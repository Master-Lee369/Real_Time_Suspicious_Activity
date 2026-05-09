# surveillance/forms.py
from django import forms
from .models import UploadedVideo


class UploadForm(forms.ModelForm):
    class Meta:
        model = UploadedVideo
        fields = ['video']
