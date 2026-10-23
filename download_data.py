"""Download the four processed UCI Heart Disease files into ./data (needs internet)."""
import urllib.request
from pathlib import Path

from cad.config import COHORT_FILES

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
out = Path("data")
out.mkdir(exist_ok=True)
for fname in COHORT_FILES.values():
    print("downloading", fname)
    urllib.request.urlretrieve(BASE + fname, out / fname)
print("done")
