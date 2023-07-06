import pandas as pd
import numpy as np

def get_frac_weights(d: float, n: int) -> np.array:
    """Generates an array of fractional weights based on differencing order d.

    Args:
        d (float): Differencing order. Must be greater than zero.
        n (int): The length of the series you want

    Returns:
        np.array: Array of the weights to apply to create your fractional series
    """
    assert d > 0, "Differencing order has to be greater than 0"
    weights = []

    for i in range(n):
        if i == 0:
            weights.append(1)
        else:
            w = -weights[i-1]*(d - i + 1)/i
            weights.append(w)

    return np.array(weights)

class FFDTransform:

    def __init__(self, d: float, tol: float) -> None:
        self.d = d
        self.tol = tol
        self.is_fit = False

    @classmethod
    def get_frac_weights(d: float, n: int) -> np.array:

        weights = []

        for i in range(n):
            if i == 0:
                weights.append(1)
            else:
                w = -weights[i-1]*(d - i + 1)/i
                weights.append(w)

        return np.array(weights)

    def get_l_star_value(self) -> int:

        abs_weights = np.abs(self.weights)
        l_star = np.where(abs_weights >= self.tol)[0].max() + 1
        return l_star

    def fit(self, series: pd.Series):

        self.weights = get_frac_weights(d = self.d, n = series.shape[0])
        self.l_star = self.get_l_star_value()
        self.is_fit = True

    def fit_transform(self, series: pd.Series) -> pd.Series:
        self.fit(series)
        return self.transform(series)

    def transform(self, series: pd.Series) -> pd.Series:
        assert self.is_fit, "The FFD has to be fit. Please call the fit method"

        if self.l_star > series.shape[0]:
            l = series.shape[0]
        else:
            l = self.l_star

        w = self.weights[:l]
        return series.rolling(l).apply(lambda x: (x*w[::-1]).sum())