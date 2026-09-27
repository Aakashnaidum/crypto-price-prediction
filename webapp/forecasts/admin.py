from django.contrib import admin

from .models import PredictionLog


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "user", "asset", "model_label", "model_forecast", "persistence_forecast")
    list_filter = ("asset",)
