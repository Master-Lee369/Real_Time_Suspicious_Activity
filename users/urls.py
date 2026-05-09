from django.urls import path
from django.contrib.auth import views as auth_views
from .views import SignUpView

urlpatterns = [
    path('register/', SignUpView.as_view(), name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    # Password reset URLs are included via contrib.auth.urls
]
