import numpy as np
import pytest

from cryptoforecast.artifacts import load_bundle, predict_next_close, save_bundle, validate_candle
from cryptoforecast.evaluation import common_frame
from cryptoforecast.features import build_dataset
from cryptoforecast.models import deployable_forecasters
from cryptoforecast.synthetic import synthetic_history


def test_bundle_roundtrip_and_prediction_consistency(tmp_path):
    frame = common_frame(build_dataset(synthetic_history(300)))
    model = deployable_forecasters()[0].fit(frame)
    path = tmp_path / "x.joblib"
    save_bundle(path, asset_key="bitcoin", model=model, trained_through="2020-10-01", evidence={"beats_persistence": False})
    bundle = load_bundle(path)
    last = frame.iloc[-1]
    out = predict_next_close(bundle, open_=last.Open, high=last.High, low=last.Low, close=last.Close, volume=last.Volume)
    # The app path (single candle) must match the training/evaluation path exactly.
    assert out["model_forecast"] == pytest.approx(model.predict_price(frame.iloc[[-1]])[0])
    assert out["persistence_forecast"] == last.Close


@pytest.mark.parametrize("candle", [
    dict(open_=10, high=9, low=8, close=9.5, volume=1),   # high below open
    dict(open_=10, high=12, low=11, close=11.5, volume=1),  # low above open
    dict(open_=0, high=12, low=0, close=11.5, volume=1),
    dict(open_=10, high=12, low=9, close=11, volume=-5),
    dict(open_=np.nan, high=12, low=9, close=11, volume=5),
])
def test_invalid_candles_rejected(candle):
    with pytest.raises(ValueError):
        validate_candle(candle["open_"], candle["high"], candle["low"], candle["close"], candle["volume"])
