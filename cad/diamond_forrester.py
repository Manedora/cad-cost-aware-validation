"""Diamond-Forrester pretest probability of angiographic CAD (Diamond & Forrester 1979).

Probability by age decade, sex and chest-pain type. Undefined for asymptomatic patients
(cp = 4) and for ages outside 30-69. VERIFY these values against Table 2 of the original
paper (reference 8 of the manuscript) before submission."""
import numpy as np

# percent: {age_decade: {cp: (men, women)}}; cp 1 typical, 2 atypical, 3 non-anginal
_TABLE = {
    30: {1: (69.7, 25.8), 2: (21.8, 4.0), 3: (5.2, 0.8)},
    40: {1: (87.3, 55.2), 2: (46.1, 13.3), 3: (14.1, 2.8)},
    50: {1: (92.0, 79.4), 2: (58.9, 32.4), 3: (21.5, 8.4)},
    60: {1: (94.3, 90.6), 2: (67.1, 54.4), 3: (28.1, 18.6)},
}


def pretest_probability(age, sex, cp):
    """Probability in [0, 1]; NaN where the score is not defined. sex: 1 = male, 0 = female."""
    if np.isnan(age) or np.isnan(sex) or np.isnan(cp):
        return np.nan
    decade = int(age // 10) * 10
    if decade not in _TABLE or int(cp) not in _TABLE[decade]:
        return np.nan
    men, women = _TABLE[decade][int(cp)]
    return (men if int(sex) == 1 else women) / 100.0


def score_for(df):
    return np.array([pretest_probability(a, s, c)
                     for a, s, c in zip(df["age"], df["sex"], df["cp"])])
