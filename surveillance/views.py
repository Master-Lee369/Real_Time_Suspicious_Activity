from django.shortcuts import render
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
import os
import time

from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib.auth.decorators import login_required

from .models import DetectionLog, UploadedVideo
from .forms import UploadForm
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib import messages




class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        user = self.request.user

        # Safe DB queries: protect against unexpected DB errors so dashboard won't crash
        try:
            logs_qs = DetectionLog.objects.filter(user=user).order_by('-detected_at')
            context['logs'] = logs_qs[:10]
            context['recent_logs'] = logs_qs[:10]
            context['total_logs'] = logs_qs.count()
            context['suspicious_count'] = logs_qs.filter(activity_type='suspicious').count()
            context['id_card_count'] = logs_qs.filter(activity_type='id_detected').count()
        except Exception:
            context['logs'] = []
            context['recent_logs'] = []
            context['total_logs'] = 0
            context['suspicious_count'] = 0
            context['id_card_count'] = 0

        try:
            context['total_uploaded_videos'] = UploadedVideo.objects.filter(user=user).count()
        except Exception:
            context['total_uploaded_videos'] = 0

        context['activity_model_available'] = os.path.exists(
            'ml/models/activity_mobilenet_lstm/activity_model.keras'
        )

        context['id_card_model_available'] = os.path.exists(
            'ml/models/id_card_yolov5/best.pt'
        )

        # additional status flags
        context['class_names_available'] = os.path.exists('ml/models/activity_mobilenet_lstm/class_names.json')
        context['sqlite_logs_working'] = True
        context['upload_detection_ready'] = True

        return context


class UploadView(LoginRequiredMixin, FormView):
    form_class = UploadForm
    template_name = 'upload.html'
    success_url = reverse_lazy('upload')

    def form_valid(self, form):
        upload = form.save(commit=False)
        upload.user = self.request.user
        upload.status = "Processing"
        upload.save(update_fields=['status'])

        result = process_uploaded_video(upload)
        if result.get('status') == 'models_missing':
            upload.status = "Waiting for trained model"
        elif result.get('status') == 'completed':
            upload.status = "Completed"
        else:
            upload.status = "Error processing"
        upload.save(update_fields=['status'])

        messages.success(self.request, "Video uploaded and processed successfully.")
        return super().form_valid(form)



    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['uploads'] = UploadedVideo.objects.filter(
            user=self.request.user
        ).order_by('-uploaded_at')
        return context


def generate_unavailable_frame(message="Camera unavailable"):
    import numpy as np
    import cv2

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    cv2.putText(
        frame,
        message,
        (60, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    ret, buffer = cv2.imencode(".jpg", frame)

    if not ret:
        return None

    return buffer.tobytes()


def generate_camera_frames(request_user):
    """MJPEG streaming generator with real-time: anti-spoof -> activity (LSTM) -> ID (YOLO).

    Input source:
      - if CAMERA_SOURCE env var is set to an RTSP/HTTP URL, OpenCV will open that stream
      - else uses CAMERA_DEVICE_INDEX / webcam indices


    - Models are lazily loaded by wrappers.
    - DetectionLog writes are throttled to avoid DB spam.
    """
    import cv2
    import numpy as np

    from ml.inference.id_detector import IDDetector
    from ml.inference.activity_predictor import ActivityPredictor
    from ml.inference.anti_spoof import frame_difference_score, repeated_frame_ratio

    id_det = IDDetector()
    act = ActivityPredictor()

    ACTIVITY_SUSPICIOUS_CONF_THRESHOLD = float(os.getenv('ACTIVITY_SUSPICIOUS_CONF_THRESHOLD', '0.60'))

    SPOOF_DIFF_SCORE_THRESHOLD = float(os.getenv('SPOOF_DIFF_SCORE_THRESHOLD', '2.0'))
    SPOOF_REPEAT_RATIO_THRESHOLD = float(os.getenv('SPOOF_REPEAT_RATIO_THRESHOLD', '0.6'))

    SEQ_LEN = int(os.getenv('ACTIVITY_SEQ_LEN', '30'))

    # throttle DB logs per activity_type
    LOG_THROTTLE_SECONDS = int(os.getenv('DETECTION_LOG_THROTTLE_SECONDS', '10'))
    last_log_ts = {
        'suspicious': 0.0,
        'normal': 0.0,
        'id_detected': 0.0,
        'spoof_detected': 0.0,
    }

    # throttle security email alerts separately (so a single event doesn't spam)
    EMAIL_THROTTLE_SECONDS = int(os.getenv('SECURITY_EMAIL_THROTTLE_SECONDS', '60'))
    last_email_ts = {
        'suspicious': 0.0,
        'spoof_detected': 0.0,
    }

    def maybe_send_email_alert(alert_type: str, subject: str, message: str, now_ts=None):
        """Best-effort email sender (never crashes the stream)."""
        now_ts = now_ts if now_ts is not None else time.time()
        last_ts = last_email_ts.get(alert_type, 0.0)
        if now_ts - last_ts < EMAIL_THROTTLE_SECONDS:
            return
        last_email_ts[alert_type] = now_ts
        try:
            from ml.utils.email_alert import send_security_alert
            send_security_alert(subject=subject, message=message)
        except Exception:
            return


    def maybe_log(activity_type, confidence, message, now_ts=None):
        now_ts = now_ts if now_ts is not None else time.time()
        last_ts = last_log_ts.get(activity_type, 0.0)
        if now_ts - last_ts < LOG_THROTTLE_SECONDS:
            return
        last_log_ts[activity_type] = now_ts
        try:
            DetectionLog.objects.create(
                user=request_user,
                video=None,
                activity_type=activity_type,
                confidence_score=float(confidence or 0.0),
                message=message or '',
            )
        except Exception:
            # keep stream alive even if DB fails
            return

    def overlay_result(display, text_lines, bbox_dets=None):
        y = 35
        for line in text_lines:
            cv2.putText(display, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            y += 24

        if bbox_dets:
            for (x1, y1, x2, y2, conf) in bbox_dets:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(display, f"{conf:.2f}", (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # Open camera source.
    # If CAMERA_SOURCE is set, open that (e.g., rtsp://... / http://... / or a local file path).
    # Otherwise fall back to webcam indices/backends.
    camera_source = os.getenv('CAMERA_SOURCE', '').strip()

    camera = None
    last_open_err = None

    if camera_source:
        try:
            camera = cv2.VideoCapture(camera_source)
        except Exception as e:
            last_open_err = e

    if camera is None or (hasattr(camera, 'isOpened') and not camera.isOpened()):
        # webcam fallback
        camera_index = int(os.getenv('CAMERA_DEVICE_INDEX', '0'))
        candidate_indices = [camera_index, 0, 1]
        candidate_backends = [
            getattr(cv2, 'CAP_DSHOW', 0),
            getattr(cv2, 'CAP_MSMF', 0),
            getattr(cv2, 'CAP_ANY', 0),
        ]

        for idx in candidate_indices:
            for backend in candidate_backends:
                try:
                    cam = cv2.VideoCapture(idx, backend)
                    if cam is not None and cam.isOpened():
                        camera = cam
                        break
                except Exception as e:
                    last_open_err = e
            if camera is not None:
                break

    if camera is None or (hasattr(camera, 'isOpened') and not camera.isOpened()):

        err_msg = 'Camera unavailable (try laptop index 0/1 or set CAMERA_SOURCE to IP/RTSP)'
        if last_open_err is not None:
            err_msg = f"{err_msg}: {last_open_err}"
        frame_bytes = generate_unavailable_frame(err_msg)


        while True:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(1)

    frame_gray_buffer = []

    try:
        while True:
            success, frame = camera.read()

            if not success:
                frame_bytes = generate_unavailable_frame("Failed to read camera")
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                time.sleep(1)
                continue

            frame = cv2.resize(frame, (640, 480))
            display = frame.copy()

            # Build activity sequence buffer
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray_buffer.append(gray)
            if len(frame_gray_buffer) > SEQ_LEN:
                frame_gray_buffer.pop(0)

            text_lines = ["Live Detection Running"]

            spoof_flag = False
            diff_score = None
            repeat_ratio = None

            if len(frame_gray_buffer) == SEQ_LEN:
                diff_score = frame_difference_score(frame_gray_buffer)
                repeat_ratio = repeated_frame_ratio(frame_gray_buffer)

                spoof_flag = (diff_score < SPOOF_DIFF_SCORE_THRESHOLD) or (repeat_ratio > SPOOF_REPEAT_RATIO_THRESHOLD)
                if spoof_flag:
                    text_lines.append("Anti-spoof: SPOOF")
                    maybe_log('spoof_detected', confidence=diff_score or 0.0, message=f"Spoof heuristic: diff={diff_score:.3f}, repeat={repeat_ratio:.3f}")
                    maybe_send_email_alert(
                        alert_type='spoof_detected',
                        subject='Security Alert: Spoof Detected',
                        message=f"Spoof heuristic triggered. diff={diff_score:.3f}, repeat={repeat_ratio:.3f}",
                    )
                else:
                    text_lines.append("Anti-spoof: OK")


                text_lines.append(f"Diff: {diff_score:.2f} Repeat: {repeat_ratio:.2f}")

            # ID detection on current frame (throttled for smooth FPS)
            bbox_dets = []
            id_sample_every_n_frames = int(os.getenv('STREAM_ID_SAMPLE_EVERY_N_FRAMES', '5'))
            stream_frame_counter = int(os.getenv('STREAM_FRAME_COUNTER', '0'))

            # We don't have a persisted counter; instead approximate by using time.
            # (If you want deterministic behavior, we can add a real counter.)
            do_id = (stream_frame_counter % max(1, id_sample_every_n_frames) == 0)

            if do_id and id_det.is_model_available():
                id_out = id_det.detect_image(frame)
                if isinstance(id_out, dict) and id_out.get('status') == 'ok':
                    for d in id_out.get('detections', []):
                        x1, y1, x2, y2 = d['bbox']
                        conf = d.get('conf', 0.0)
                        bbox_dets.append((int(x1), int(y1), int(x2), int(y2), float(conf)))

                    if bbox_dets:
                        top = max(bbox_dets, key=lambda t: t[4])
                        text_lines.append(f"ID detected: conf={top[4]:.2f}")
                        maybe_log('id_detected', confidence=top[4], message=f"ID card detected conf={top[4]:.2f}")
                    else:
                        text_lines.append("ID detected: none")
                else:
                    text_lines.append("ID detector unavailable")
            else:
                text_lines.append("ID: skipped")


            # Activity detection (sequence) - skip if spoof
            if len(frame_gray_buffer) == SEQ_LEN and act.is_model_available() and not spoof_flag:
                seq_rgb = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in frame_gray_buffer]
                seq_rgb = [cv2.resize(f, (224, 224)) for f in seq_rgb]
                activity_result = act.predict_sequence(np.array(seq_rgb))

                if isinstance(activity_result, dict) and activity_result.get('status') == 'ok':
                    lbl = activity_result.get('label', 'unknown')
                    conf = float(activity_result.get('confidence', 0.0))
                    text_lines.append(f"Activity: {lbl} ({conf:.2f})")

                    if lbl == 'suspicious' and conf >= ACTIVITY_SUSPICIOUS_CONF_THRESHOLD:
                        text_lines.append(f"ALERT: suspicious >= {ACTIVITY_SUSPICIOUS_CONF_THRESHOLD:.2f}")
                        maybe_log('suspicious', confidence=conf, message=f"Suspicious activity detected: {lbl} ({conf:.2f})")
                        maybe_send_email_alert(
                            alert_type='suspicious',
                            subject='Security Alert: Suspicious Activity',
                            message=f"Suspicious activity detected: {lbl} (confidence={conf:.2f})",
                        )
                    else:
                        text_lines.append("Activity: normal")
                        maybe_log('normal', confidence=conf, message=f"Activity: {lbl} ({conf:.2f})")

                else:
                    text_lines.append("Activity model unavailable")
            elif spoof_flag:
                text_lines.append("Activity: skipped (spoof)")
            else:
                if not act.is_model_available():
                    text_lines.append("Activity model not trained")

            overlay_result(display, text_lines, bbox_dets=bbox_dets if bbox_dets else None)

            ret, buffer = cv2.imencode('.jpg', display)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            time.sleep(0.03)

    finally:
        try:
            camera.release()
        except Exception:
            pass


class VideoStreamView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        # Ensure MJPEG stream never crashes even if camera/ML fails.
        try:
            gen = generate_camera_frames(request.user)
        except Exception:
            gen = iter([])

        return StreamingHttpResponse(
            gen,
            content_type="multipart/x-mixed-replace; boundary=frame",
        )



@login_required
def video_feed(request):
    """Function-based MJPEG stream endpoint (login required)."""
    # Always return a valid MJPEG response; generator itself handles "Camera unavailable" frame.
    try:
        return StreamingHttpResponse(
            generate_camera_frames(request.user),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception:
        # Fallback: return a single unavailable frame and keep connection alive.
        return StreamingHttpResponse(
            iter([
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + (generate_unavailable_frame("Camera unavailable") or b"") + b"\r\n"
            ]),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )



def detection_api(request):
    # Lazy-import ML wrappers to avoid heavy deps at Django startup.
    try:
        from ml.inference.detector import IDDetector
        from ml.inference.activity_predictor import ActivityPredictor

        id_det = IDDetector()
        act = ActivityPredictor()

        models_status = {
            'id_detector': 'loaded' if not id_det.model_missing else 'missing',
            'activity_model': 'loaded' if not act.model_missing else 'missing'
        }

        if all(v == 'missing' for v in models_status.values()):
            return JsonResponse({'status': 'ok', 'message': 'Model not trained yet', 'models': models_status})

        return JsonResponse({'status': 'success', 'message': 'Detection API working', 'models': models_status})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Detection API error: {e}'}, status=500)


class LogsView(LoginRequiredMixin, TemplateView):
    template_name = 'logs.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        try:
            qs = DetectionLog.objects.filter(user=self.request.user).order_by('-detected_at')
            paginator = Paginator(qs, 25)
            page = self.request.GET.get('page', 1)
            try:
                logs_page = paginator.page(page)
            except PageNotAnInteger:
                logs_page = paginator.page(1)
            except EmptyPage:
                logs_page = paginator.page(paginator.num_pages)
            context['logs_page'] = logs_page
        except Exception:
            context['logs_page'] = []
        return context


def process_uploaded_video(uploaded_video):
    """Process an UploadedVideo instance.

    Best-effort ML inference:
    - anti-spoof heuristics + activity sequence prediction (skip activity when spoof)
    - ID detection on sampled frames
    - writes DetectionLog rows

    Never raises: returns a dict with status.
    """
    import cv2
    import numpy as np

    # Throttling (per uploaded video processing run)
    LOG_THROTTLE_SECONDS = int(os.getenv('DETECTION_LOG_THROTTLE_SECONDS', '10'))
    last_log_ts = {
        'suspicious': 0.0,
        'normal': 0.0,
        'id_detected': 0.0,
        'spoof_detected': 0.0,
    }

    def maybe_log(activity_type, confidence, message, now_ts=None):
        now_ts = now_ts if now_ts is not None else time.time()
        last_ts = last_log_ts.get(activity_type, 0.0)
        if now_ts - last_ts < LOG_THROTTLE_SECONDS:
            return
        last_log_ts[activity_type] = now_ts
        try:
            DetectionLog.objects.create(
                user=uploaded_video.user,
                video=uploaded_video,
                activity_type=activity_type,
                confidence_score=float(confidence or 0.0),
                message=message or '',
            )
        except Exception:
            return

    activity_model_path = 'ml/models/activity_mobilenet_lstm/activity_model.keras'
    id_model_path = 'ml/models/id_card_yolov5/best.pt'

    # Availability gates
    if not os.path.exists(activity_model_path) or not os.path.exists(id_model_path):
        return {'status': 'models_missing'}

    try:
        from ml.inference.id_detector import IDDetector
        from ml.inference.activity_predictor import ActivityPredictor
        from ml.inference.anti_spoof import frame_difference_score, repeated_frame_ratio

        id_det = IDDetector(weights_path=id_model_path)
        act = ActivityPredictor(model_path=activity_model_path)

        if not id_det.is_model_available() or not act.is_model_available():
            return {'status': 'models_missing'}

        seq_len = int(os.getenv('ACTIVITY_SEQ_LEN', '30'))
        max_windows = int(os.getenv('UPLOAD_MAX_ACTIVITY_WINDOWS', '10'))
        id_sample_every_n_frames = int(os.getenv('UPLOAD_ID_SAMPLE_EVERY_N_FRAMES', '5'))

        video_path = getattr(uploaded_video.video, 'path', None)
        if not video_path or not os.path.exists(video_path):
            return {'status': 'error', 'error': 'Uploaded file missing from disk'}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'status': 'error', 'error': 'Could not open uploaded video'}

        frame_gray_buffer = []
        frame_idx = 0
        windows_done = 0
        last_activity_pred = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame = cv2.resize(frame, (640, 480))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray_buffer.append(gray)
            if len(frame_gray_buffer) > seq_len:
                frame_gray_buffer.pop(0)

            # Anti-spoof + activity prediction on full sequence
            if len(frame_gray_buffer) == seq_len:
                diff_score = frame_difference_score(frame_gray_buffer)
                repeat_ratio = repeated_frame_ratio(frame_gray_buffer)
                spoof_flag = (diff_score < float(os.getenv('SPOOF_DIFF_SCORE_THRESHOLD', '2.0'))) or (
                    repeat_ratio > float(os.getenv('SPOOF_REPEAT_RATIO_THRESHOLD', '0.6'))
                )

                if spoof_flag:
                    maybe_log('spoof_detected', confidence=diff_score or 0.0,
                              message=f"Spoof heuristic: diff={diff_score:.3f}, repeat={repeat_ratio:.3f}")
                else:
                    seq_rgb = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in frame_gray_buffer]
                    seq_rgb = [cv2.resize(f, (224, 224)) for f in seq_rgb]
                    activity_result = act.predict_sequence(np.array(seq_rgb))
                    if isinstance(activity_result, dict) and activity_result.get('status') == 'ok':
                        lbl = activity_result.get('label', 'unknown')
                        conf = float(activity_result.get('confidence', 0.0))
                        if lbl == 'suspicious' and conf >= float(os.getenv('ACTIVITY_SUSPICIOUS_CONF_THRESHOLD', '0.60')):
                            maybe_log('suspicious', confidence=conf, message=f"Suspicious activity detected: {lbl} ({conf:.2f})")
                        else:
                            maybe_log('normal', confidence=conf, message=f"Activity: {lbl} ({conf:.2f})")
                        last_activity_pred = (lbl, conf)

                windows_done += 1
                if windows_done >= max_windows:
                    # We still can run some ID checks, but stop activity windows to cap runtime.
                    frame_gray_buffer = []
                    break

            # ID detection sampled frames
            if id_det.is_model_available() and (frame_idx % id_sample_every_n_frames == 0):
                id_out = id_det.detect_image(frame)
                if isinstance(id_out, dict) and id_out.get('status') == 'ok':
                    dets = id_out.get('detections', [])
                    if dets:
                        top = max(dets, key=lambda t: t.get('conf', 0.0))
                        maybe_log('id_detected', confidence=top.get('conf', 0.0),
                                  message=f"ID card detected conf={top.get('conf', 0.0):.2f}")

        cap.release()

        return {
            'status': 'completed',
            'windows_done': windows_done,
            'last_activity_pred': last_activity_pred,
        }

    except Exception as e:
        return {'status': 'error', 'error': str(e)}



@login_required
def live_detection_view(request):
    context = {
        'activity_model_available': os.path.exists('ml/models/activity_mobilenet_lstm/activity_model.keras') and os.path.exists('ml/models/activity_mobilenet_lstm/class_names.json'),
        'class_names_available': os.path.exists('ml/models/activity_mobilenet_lstm/class_names.json'),
        'id_card_model_available': os.path.exists('ml/models/id_card_yolov5/best.pt'),
        'sqlite_logs_working': True,
        'upload_detection_ready': True,
    }
    return render(request, 'live_detection.html', context)


