### Data Preparation

* NDVI time series per plot
* Yield (GYLD) at the end of the season per plot
* Each plot = one sample for the model.
  * Input: NDVI sequence over time. 
  * Output: Yield (a scalar).


Data needs to be reshaped
* X: shape (n_plots, n_timesteps, n_features)
    * n_plots = number of wheat plots.
      * n_timesteps = number of NDVI measurements per plot (no of days between April 27, 2024 - July 15, 2024).
      * n_features = number of VI types.

* y: shape (n_plots) = yield values.

### Reshape the dataset

* Each plot’s NDVI sequence becomes a row.
* Fill missing dates (forward fill or interpolation).
* Normalize NDVI values (0–1)


### Build LSTM Network


