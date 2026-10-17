# Issue: cart_total produces nonsensical totals for extreme inputs

## Description
`cart_total()` in `cart.py` multiplies `price_cents * quantity` with no upper-bound
check. For large quantities (e.g. 10**9 items at 10**9 cents each) the function
returns an astronomically large number instead of rejecting the input or
saturating at a reasonable maximum.

## Expected behavior
- Quantities above a sane threshold (e.g. 1_000_000) should be rejected or capped.
- The function should never return a total larger than, say, 10**12 cents.

## Reproduction
```python
from cart import cart_total
print(cart_total([(10**9, 10**9)]))  # returns 10**18 — absurd
```
