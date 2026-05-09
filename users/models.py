# users/models.py
from django.conf import settings
from django.db import models
from django.contrib.auth.models import AbstractUser



class Profile(models.Model):
    # Link Profile one-to-one with User【59†L74-L83】
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    phone = models.CharField(max_length=20, blank=True)
    department = models.CharField(max_length=50, blank=True)

    def __str__(self):
        return f"{self.user.username}'s profile"
