import math

from django import template

register = template.Library()


@register.filter
def num(value, digits=3):
    """Format a float, showing an em dash for missing/NaN values."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v):
        return "—"
    return f"{v:.{int(digits)}f}"
