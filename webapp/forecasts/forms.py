from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from cryptoforecast.artifacts import validate_candle


class CandleForm(forms.Form):
    """One completed daily candle (UTC day), as reported by an exchange/aggregator."""

    open_price = forms.FloatField(label="Open", min_value=0)
    high_price = forms.FloatField(label="High", min_value=0)
    low_price = forms.FloatField(label="Low", min_value=0)
    close_price = forms.FloatField(label="Close", min_value=0)
    volume = forms.FloatField(label="Volume", min_value=0)

    def clean(self):
        data = super().clean()
        if self.errors:
            return data
        try:
            validate_candle(data["open_price"], data["high_price"], data["low_price"],
                            data["close_price"], data["volume"])
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc
        return data


class RegisterForm(UserCreationForm):
    class Meta:
        model = User
        fields = ["username", "password1", "password2"]
