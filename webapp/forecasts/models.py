from django.conf import settings
from django.db import models

from cryptoforecast.config import ASSETS

ASSET_CHOICES = [(k, a.name) for k, a in ASSETS.items()]


class PredictionLog(models.Model):
    """A forecast a signed-in user requested. Visible only to that user."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="predictions")
    asset = models.CharField(max_length=16, choices=ASSET_CHOICES)
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    close_price = models.FloatField()
    volume = models.FloatField()
    model_label = models.CharField(max_length=64)
    model_forecast = models.FloatField()
    persistence_forecast = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.asset} {self.model_forecast:.6g} ({self.created_at:%Y-%m-%d})"
