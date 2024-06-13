
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('predict_premium', views.predict_premium, name='predict_premium'),
]
