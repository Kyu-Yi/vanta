"""cart — shopping cart total calculation.

BUG: integer overflow for large carts. In CPython ints are arbitrary precision
so this won't actually overflow — but the *intent* of the bug is that the
function silently produces wrong totals when item counts exceed realistic
bounds (e.g. a quantity of 10**9 items priced at 10**9 cents).

The fix is to add bounds checking / saturation instead of silently computing
an astronomically large (or in fixed-width languages, wrapped) total.
"""


def cart_total(items):
    """Return the total price in cents for a list of (price_cents, quantity) tuples.

    BUG: no upper-bound check. A malicious or buggy caller can pass quantities
    that produce nonsensical totals. The fix should cap or validate.
    """
    total = 0
    for price_cents, quantity in items:
        total += price_cents * quantity
    return total
