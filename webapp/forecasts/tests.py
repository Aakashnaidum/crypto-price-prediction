import json
import tempfile
from pathlib import Path

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse

from cryptoforecast.artifacts import save_bundle
from cryptoforecast.evaluation import common_frame
from cryptoforecast.features import build_dataset
from cryptoforecast.models import deployable_forecasters
from cryptoforecast.synthetic import synthetic_history

from . import services
from .models import PredictionLog

TMP = Path(tempfile.mkdtemp())
MODELS = TMP / "models"
RESULTS = TMP / "metrics.json"

VALID = {"open_price": "100", "high_price": "110", "low_price": "95", "close_price": "105", "volume": "12345"}


def setUpModule():
    frame = common_frame(build_dataset(synthetic_history(300)))
    model = deployable_forecasters()[0].fit(frame)
    evidence = {"test_period": "2020-08-01 to 2020-10-01", "n_test": 60, "mae_vs_persistence": 1.02,
                "beats_persistence": False, "dm_p_value": 0.4, "directional_accuracy": 0.49}
    for key in ("bitcoin", "ethereum"):
        save_bundle(MODELS / f"{key}.joblib", asset_key=key, model=model, trained_through="2020-10-27", evidence=evidence)
    RESULTS.write_text(json.dumps({"generated_utc": "test", "assets": {}}))


@override_settings(CRYPTO_MODELS_DIR=MODELS, CRYPTO_RESULTS_FILE=RESULTS, CRYPTO_ENABLE_LIVE_DATA=False)
class ForecastViewTests(TestCase):
    def setUp(self):
        services.get_bundle.cache_clear()
        services.get_results.cache_clear()

    def test_home_lists_assets_and_missing_models(self):
        r = self.client.get(reverse("home"))
        self.assertContains(r, "Bitcoin")
        self.assertContains(r, "Model not trained yet")  # e.g. XRP has no bundle in this test

    def test_predict_shows_model_and_persistence(self):
        r = self.client.post(reverse("predict", args=["bitcoin"]), VALID)
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Persistence baseline")
        self.assertContains(r, "did <strong>not</strong> beat persistence")
        self.assertIsNotNone(r.context["result"])
        self.assertEqual(r.context["result"]["persistence_forecast"], 105.0)
        self.assertEqual(PredictionLog.objects.count(), 0)  # anonymous forecasts are not stored

    def test_invalid_candle_rejected(self):
        bad = dict(VALID, low_price="120")
        r = self.client.post(reverse("predict", args=["bitcoin"]), bad)
        self.assertIsNone(r.context["result"])
        self.assertContains(r, "Low &lt;= min(Open, Close)")

    def test_non_numeric_input_rejected(self):
        r = self.client.post(reverse("predict", args=["bitcoin"]), dict(VALID, volume="lots"))
        self.assertIsNone(r.context["result"])

    def test_unknown_asset_404_and_untrained_asset_503(self):
        self.assertEqual(self.client.get(reverse("predict", args=["dogecoin"])).status_code, 404)
        self.assertEqual(self.client.get(reverse("predict", args=["xrp"])).status_code, 503)

    def test_history_requires_login_and_is_per_user(self):
        self.assertEqual(self.client.get(reverse("history")).status_code, 302)
        alice = User.objects.create_user("alice", password="s3cure-pass-123")
        User.objects.create_user("bob", password="s3cure-pass-123")
        self.client.force_login(alice)
        self.client.post(reverse("predict", args=["bitcoin"]), VALID)
        self.assertEqual(PredictionLog.objects.filter(user=alice).count(), 1)
        self.client.logout()
        self.client.login(username="bob", password="s3cure-pass-123")
        r = self.client.get(reverse("history"))
        self.assertContains(r, "No forecasts yet")

    def test_register_and_results_pages(self):
        r = self.client.post(reverse("register"), {"username": "carol", "password1": "Zx9!long-password", "password2": "Zx9!long-password"})
        self.assertEqual(r.status_code, 302)
        self.assertTrue(User.objects.filter(username="carol").exists())
        self.assertEqual(self.client.get(reverse("results")).status_code, 200)
