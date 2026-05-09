from django.urls import path
from .views import DashboardView, UploadView, VideoStreamView, detection_api

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('upload/', UploadView.as_view(), name='upload'),
    path('stream/', VideoStreamView.as_view(), name='video_feed'),
    path('api/detect/', detection_api, name='detection_api'),
]
