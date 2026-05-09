from django.contrib import admin
from .models import UploadedVideo, DetectionLog


@admin.register(UploadedVideo)
class UploadedVideoAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'video', 'status', 'uploaded_at')
    list_filter = ('status', 'uploaded_at')
    search_fields = ('user__username', 'video')


@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'activity_type', 'confidence_score', 'detected_at')
    list_filter = ('activity_type', 'detected_at')
    search_fields = ('user__username', 'activity_type', 'message')