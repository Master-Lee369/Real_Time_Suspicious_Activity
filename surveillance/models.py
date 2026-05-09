from django.db import models
from django.contrib.auth.models import User


class UploadedVideo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.FileField(upload_to='uploaded_videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=50, default='Pending')

    def __str__(self):
        return f"{self.user.username} - {self.video.name}"


class DetectionLog(models.Model):
    ACTIVITY_CHOICES = [
        ('normal', 'Normal'),
        ('suspicious', 'Suspicious'),
        ('id_detected', 'ID Detected'),
        ('spoof_detected', 'Spoof Detected'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.ForeignKey(
        UploadedVideo,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    activity_type = models.CharField(max_length=50, choices=ACTIVITY_CHOICES)
    confidence_score = models.FloatField(default=0.0)
    message = models.TextField(blank=True, null=True)
    detected_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.activity_type} - {self.confidence_score}"