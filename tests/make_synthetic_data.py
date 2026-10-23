"""Write SYNTHETIC files in the UCI format, only to smoke-test the code without the real data.
The numbers produced from these files are meaningless and must never be reported."""
import sys
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "data_synthetic")
out.mkdir(exist_ok=True)
spec = {"cleveland": (303, .46, False), "hungarian": (294, .36, False),
        "switzerland": (123, .93, True), "va": (200, .75, True)}
for name, (n, prev, zeros) in spec.items():
    num = (rng.random(n) < prev).astype(int) * rng.integers(1, 5, n)
    d = (num > 0).astype(float)
    age = np.clip(rng.normal(52 + 5 * d, 9, n), 28, 77).round()
    sex = (rng.random(n) < .6 + .2 * d).astype(int)
    cp = np.where(d > 0, rng.choice([1, 2, 3, 4], n, p=[.05, .1, .15, .7]),
                  rng.choice([1, 2, 3, 4], n, p=[.1, .25, .3, .35]))
    trestbps = rng.normal(130, 17, n).round()
    chol = rng.normal(245, 50, n).round()
    fbs = (rng.random(n) < .15).astype(int)
    restecg = rng.choice([0, 1, 2], n)
    thalach = np.clip(rng.normal(150 - 20 * d, 20, n), 70, 200).round()
    exang = (rng.random(n) < .2 + .4 * d).astype(int)
    oldpeak = np.clip(rng.gamma(1.2, .6 + .8 * d, n), 0, 6).round(1)
    cols = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
            rng.choice([1, 2, 3], n), rng.choice([0, 1, 2, 3], n), rng.choice([3, 6, 7], n), num]
    arr = np.column_stack(cols).astype(object)
    miss = rng.random(arr.shape) < (.25 if zeros else .03)
    miss[:, [0, 1, 2, 13]] = False
    arr[miss] = "?"
    if zeros:                       # zero = not recorded; Swiss cholesterol never measured
        arr[(rng.random(n) < .1), 3] = 0
        arr[:, 4] = 0 if name == "switzerland" else arr[:, 4]
    with open(out / f"processed.{name}.data", "w") as f:
        for row in arr:
            f.write(",".join(str(v) if not isinstance(v, float) else f"{v:g}" for v in row) + "\n")
print("synthetic data written to", out)
