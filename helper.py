"""
@author: Lochana Marasinghe
@date: 10/17/2025
@description: 
"""
from typing import Tuple

import numpy as np


def make_windows(dataframe, features, output_variable: str, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for plot_id, group in dataframe.groupby('plot_id'):
        feature_matrix = group[features].values
        yield_value = group[output_variable].iloc[0]
        for i in range(len(feature_matrix) - window_size + 1):
            window = feature_matrix[i:i + window_size, :]
            X_list.append(window)
            y_list.append(yield_value)
    return np.array(X_list), np.array(y_list).reshape(-1, 1)


def make_progressive_windows(dataframe, features, output_variable: str, window_size: int, pad_value: int) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, lengths_list, no_days = [], [], [], []
    for plot_id, group in dataframe.groupby('plot_id'):
        print("Processing plot {}".format(plot_id))
        group = group.sort_values('date')
        assert group['date'].is_monotonic_increasing, f"Plot {plot_id} not sorted!"

        feature_matrix = group[features].values.astype(np.float32)
        yield_value = group[output_variable].iloc[0].astype(np.float32)

        # Progressive prediction: from day 1 to full season
        for i in range(1, len(feature_matrix) + 1):  # from 1 day to full length
            window = feature_matrix[:i, :]  # NDVI from day 1 to day i
            assert window.shape[0] > 0
            # pad shorter sequences (for model input consistency)
            if len(window) < window_size:
                pad_len = window_size - len(window)
                pad = np.full((pad_len, feature_matrix.shape[1]), pad_value, dtype=np.float32)
                # mask = np.concatenate([np.zeros(pad_len), np.ones(len(window))])
                window = np.vstack([pad, window])  # pad at the beginning
            else:
                pad_len = 0
                window = window[-window_size:, :]  # use last window_size days
                # mask = np.ones(window_size)

            X_list.append(window)
            y_list.append(yield_value)
            lengths_list.append(window_size - pad_len)
            no_days.append(i)

    X_array = np.array(X_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    lengths_array = np.array(lengths_list, dtype=np.int64)
    no_days_array = np.array(no_days, dtype=np.int64)

    return X_array, y_array, lengths_array, no_days_array


def filter_window_data_by_length(X, y, valid_lengths, no_days, needed_day):
    X = np.array(X)
    y = np.array(y)
    valid_lengths = np.array(valid_lengths)
    no_days = np.array(no_days)

    mask = no_days == needed_day

    return X[mask], y[mask], valid_lengths[mask]


def filter_window_by_length_range(X, y, valid_lengths, no_days, days_min, days_max):
    X = np.array(X)
    y = np.array(y)
    valid_lengths = np.array(valid_lengths)
    no_days = np.array(no_days)

    mask = np.ones(len(no_days), dtype=bool)
    mask &= (no_days >= days_min)
    mask &= (no_days <= days_max)

    return X[mask], y[mask], valid_lengths[mask]