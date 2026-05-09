from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from surveillance import views
from django.urls import include
from users import views as user_views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('accounts/', include('users.urls')),
    path('accounts/', include('django.contrib.auth.urls')),

    path('', include('surveillance.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)