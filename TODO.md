# TODO - Suspicious Activity System Upgrade

- [x] Step 1: Wire ML inference + anti-spoofing + SQLite logging into live MJPEG stream (`surveillance/views.py`).
- [ ] Step 2: Add throttling so DB logs/email alerts don’t spam (`surveillance/views.py` + helper if needed).
- [ ] Step 3: Update live stream overlays to display ID detection + activity label/confidence + spoof status.
- [ ] Step 4: Confirm model availability checks and ensure stream still works when models are missing.
- [ ] Step 5: After live stream works, add email alerting (env/SMTP config) and trigger on suspicious detections.
- [ ] Step 6: Implement uploaded video processing (extract frames/sequences, run inference, write DetectionLog, update UploadedVideo.status).
- [ ] Step 7: Harden `RealTimeSecurity/settings.py` (DEBUG/SECRET_KEY/ALLOWED_HOSTS via env, improve security headers/cookies).
- [ ] Step 8: Improve detection_api (health/inference), and add smoke tests.

# Implementation tracking for approved continuation
- [x] Continue-Plan A: Steps 2+3+6 (email throttling in stream + real uploaded-video processing, sync call from UploadView)


