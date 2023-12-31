# -*- coding: utf-8 -*-
from __future__ import unicode_literals, absolute_import

from django.urls import path, include
from django.contrib import admin
from app.example.urls import urlpatterns as example_urls

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("django_river_ml.urls", namespace="django_river_ml")),
]

urlpatterns += example_urls
