import numpy as np
import pywt
from scipy.stats import skew, kurtosis
from sklearn.base import BaseEstimator, TransformerMixin


class DWTFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db6', level=5):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def extract_features(signal):
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
            features = []
            for c in coeffs:
                features.extend(
                    [np.mean(c),
                     np.std(c),
                     np.max(c),
                     np.min(c),
                     np.sum(np.abs(c) ** 2)]
                )
            return features

        return np.array([extract_features(x) for x in X])


class TimeDomainFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def extract_features(signal):
            return [
                np.mean(signal),
                np.std(signal),
                skew(signal),
                kurtosis(signal)
            ]

        return np.array([extract_features(x) for x in X])
