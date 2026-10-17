`shipping_band(1000)` should return `"standard"`, not `"heavy"`.

Right now the threshold logic treats weights strictly below `1000` grams as
`"standard"` and everything else as `"heavy"`, so the exact `1000` gram edge
falls into the wrong band.

Please fix that boundary bug without changing the existing behavior at the
other thresholds.
