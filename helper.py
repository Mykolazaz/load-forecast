import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf

def plot_ccf(y, X, max_lags=None, n_lags=14):
    
    y = np.asarray(y)
    n = len(y)
    conf = 1.96 / np.sqrt(n)
    
    for col in X.columns:
        x = np.asarray(X[col])

        ccf_values = ccf(x, y)

        if max_lags is not None:
            ccf_values = ccf_values[:max_lags]
            lags = np.arange(max_lags)
        else:
            lags = np.arange(len(ccf_values))

        plt.figure(figsize=(8, 4))
        plt.stem(lags, ccf_values)
        plt.axhline(conf, linestyle="--", color="gray")
        plt.axhline(-conf, linestyle="--", color="gray")
        plt.axhline(0, color="black")

        plt.xlabel("Lag")
        plt.ylabel("CCF")
        plt.title(f"{col} leading y")
        plt.show()

        significant_lags = np.where(np.abs(ccf_values) > conf)[0]

        for lag in significant_lags[:n_lags + 1]:
            print(f"Lag {lag}: {ccf_values[lag]:.3f}")