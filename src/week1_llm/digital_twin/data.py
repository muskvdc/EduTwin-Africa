
from __future__ import annotations

import numpy as np


ACTIVITY_NAMES = {
    0: "Sleep",
    1: "Work",
    2: "Exercise",
    3: "Social",
    4: "Leisure",
}


def generate_synthetic_behavioral_data(
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the Week 1 Digital Twin-style feature set.

    Features:
        sleep_hours, mood, energy, day_of_week, time_of_day

    Labels:
        Sleep, Work, Exercise, Social, Leisure

    This is synthetic educational data, not real user data.
    """
    rng = np.random.default_rng(seed)
    sleep = rng.uniform(4.5, 9.0, n_samples)
    mood = rng.uniform(0.0, 1.0, n_samples)
    energy = rng.uniform(0.0, 1.0, n_samples)
    day = rng.integers(0, 7, n_samples)
    time = rng.uniform(0.0, 24.0, n_samples)

    y = np.empty(n_samples, dtype=int)

    for i in range(n_samples):
        if time[i] < 6 or time[i] >= 23:
            label = 0  # Sleep
        elif energy[i] > 0.72 and 6 <= time[i] < 10:
            label = 2  # Exercise
        elif 8 <= time[i] < 17 and energy[i] > 0.35:
            label = 1  # Work
        elif 17 <= time[i] < 22 and mood[i] > 0.55:
            label = 3  # Social
        else:
            label = 4  # Leisure

        # Add a small amount of behavioral noise.
        if rng.random() < 0.12:
            label = int(rng.integers(0, 5))
        y[i] = label

    X = np.column_stack([sleep / 9.0, mood, energy, day / 6.0, time / 24.0])
    return X.astype(np.float32), y
