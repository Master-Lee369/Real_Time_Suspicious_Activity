from django.urls import path
from .views import DashboardView, UploadView, VideoStreamView, detection_api, LogsView, video_feed, live_detection_view

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('upload/', UploadView.as_view(), name='upload'),
    # page that shows live detection UI
    path('stream/', live_detection_view, name='stream_page'),
    # MJPEG video feed endpoint (used by <img src> tags)
    path('video-feed/', video_feed, name='video_feed'),
    path('logs/', LogsView.as_view(), name='logs'),
    path('api/detect/', detection_api, name='detection_api'),
]
