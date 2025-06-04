import numpy as np


def shuffle(market_data: np.ndarray, market_features: np.ndarray, axis=0):
    assert market_data.shape[axis] == market_features.shape[axis]

    indices = np.random.permutation(market_data.shape[axis])
    market_data_shuffled = market_data[indices]
    market_features_shuffled = market_features[indices]

    return market_data_shuffled, market_features_shuffled
