from django.shortcuts import render
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse

from .models import DetectionLog
from .forms import UploadForm


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context['logs'] = DetectionLog.objects.filter(
            user=self.request.user
        ).order_by('-detected_at')

        context['upload_form'] = UploadForm()

        return context


class UploadView(LoginRequiredMixin, FormView):
    form_class = UploadForm
    template_name = 'upload.html'
    success_url = reverse_lazy('dashboard')

    def form_valid(self, form):
        upload = form.save(commit=False)
        upload.user = self.request.user
        upload.save()

        return super().form_valid(form)


class VideoStreamView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        return render(request, 'stream.html')


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