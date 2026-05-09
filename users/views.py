# users/views.py
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import CreateView
from django.shortcuts import redirect
from .forms import SignUpForm

class SignUpView(CreateView):
    form_class = SignUpForm
    template_name = 'registration/register.html'
    success_url = reverse_lazy('login')
