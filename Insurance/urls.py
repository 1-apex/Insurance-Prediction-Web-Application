
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("admin", admin.site.urls),
    path('', views.index, name='index'),
    path('predict_premium', views.predict_premium, name='predict_premium')
]
