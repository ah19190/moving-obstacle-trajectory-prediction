import numpy as np

def AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=True, add_correction=True):
    """
    Calculates the Akaike Information Criterion (AIC) for a given model.
    """
    rss = np.sum((x_dot_valid - x_dot_pred) ** 2,
                 axis=0 if keep_dimensionalized else None)
    m = x_dot_valid.shape[0] * (1 if keep_dimensionalized else x_dot_valid.shape[1])
    aic = 2 * k + m * np.log(rss / m)
    if add_correction:
        correction_term = (2 * (k + 1) * (k + 2)) / max(m - k - 2, 1)  # In case k == m
        aic += correction_term
    return aic