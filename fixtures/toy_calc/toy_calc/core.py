def clamp(value, low, high):
    if low > high:
        raise ValueError("low cannot be greater than high")
    return max(low, min(value, high))


def shipping_band(weight_grams):
    if weight_grams < 0:
        raise ValueError("weight_grams must be non-negative")
    if weight_grams < 500:
        return "light"
    if weight_grams < 1000:
        return "standard"
    return "heavy"


def apply_discount(total_cents, percent):
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return round(total_cents * (100 - percent) / 100)


def arithmetic_mean(values):
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)
