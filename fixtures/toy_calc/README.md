# toy_calc fixture

This fixture is a tiny pure-Python repo for the first controlled end-to-end
Vanta runs.

The planted bug is in `shipping_band()`:

- expected: `shipping_band(1000) == "standard"`
- current buggy behavior: `shipping_band(1000) == "heavy"`

The existing pytest suite passes, but it does not cover the `500g` or `1000g`
boundaries.

Expected outcomes:

1. Review mode with `bad.patch`
   - Vanta should reject it.
   - The hidden regression is that it changes `shipping_band(500)` from
     `"standard"` to `"light"`.

2. Review mode with `good.patch`
   - Vanta should accept it.

3. Repair mode on the original buggy repo
   - Vanta should produce a patch that ends up accepted.

Backend matrix:

- force subprocess: `VANTA_SANDBOX_BACKEND=subprocess`
- force shuru: `VANTA_SANDBOX_BACKEND=shuru`

Convenience runner:

- `./fixtures/toy_calc/run_matrix.sh subprocess`
- `./fixtures/toy_calc/run_matrix.sh shuru`
- `./fixtures/toy_calc/run_matrix.sh all`
