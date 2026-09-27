from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import redirect, render

from cryptoforecast.artifacts import predict_next_close
from cryptoforecast.config import ASSETS

from .forms import CandleForm, RegisterForm
from .models import PredictionLog
from .services import ModelUnavailable, get_bundle, get_results, latest_completed_candle


def _asset_or_404(asset_key):
    if asset_key not in ASSETS:
        raise Http404("Unknown asset")
    return ASSETS[asset_key]


def home(request):
    cards = []
    for key, asset in ASSETS.items():
        try:
            bundle = get_bundle(key)
            cards.append({"asset": asset, "bundle": bundle, "evidence": bundle["evidence"]})
        except ModelUnavailable:
            cards.append({"asset": asset, "bundle": None, "evidence": None})
    return render(request, "forecasts/home.html", {"cards": cards})


def predict(request, asset_key):
    asset = _asset_or_404(asset_key)
    try:
        bundle = get_bundle(asset_key)
    except ModelUnavailable as exc:
        return render(request, "forecasts/predict.html", {"asset": asset, "model_error": str(exc)}, status=503)

    live = None
    result = None
    if request.method == "POST":
        form = CandleForm(request.POST)
        if form.is_valid():
            c = form.cleaned_data
            result = predict_next_close(
                bundle, open_=c["open_price"], high=c["high_price"], low=c["low_price"],
                close=c["close_price"], volume=c["volume"],
            )
            if request.user.is_authenticated:
                PredictionLog.objects.create(
                    user=request.user, asset=asset_key, model_label=bundle["model_label"],
                    model_forecast=result["model_forecast"],
                    persistence_forecast=result["persistence_forecast"], **c,
                )
    else:
        live = latest_completed_candle(asset_key)
        form = CandleForm(initial={k: v for k, v in (live or {}).items() if k != "date"})

    return render(request, "forecasts/predict.html", {
        "asset": asset, "form": form, "bundle": bundle, "evidence": bundle["evidence"],
        "result": result, "live": live,
    })


def results(request):
    data = get_results()
    rows = []
    if data:
        for key, r in data["assets"].items():
            models = sorted(r["holdout"]["models"].items(), key=lambda kv: kv[1]["mae_vs_persistence"])
            rows.append({"asset": ASSETS[key], "holdout": r["holdout"], "models": models,
                         "walk_forward": r["walk_forward"], "quality": r["data_quality"]})
    return render(request, "forecasts/results.html", {"data": data, "rows": rows})


@login_required
def history(request):
    logs = PredictionLog.objects.filter(user=request.user)[:200]
    return render(request, "forecasts/history.html", {"logs": logs})


def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Account created.")
            return redirect("home")
    else:
        form = RegisterForm()
    return render(request, "registration/register.html", {"form": form})
