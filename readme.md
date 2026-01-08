### Data Preparation

* VI time series per plot
* Yield (GYLD) at the end of the season per plot
* Each plot = one sample for the model.
  * Input: VIs sequence over time. 
  * Output: Yield (a scalar).



### GRU Model Hyperparameter Tuning Results

This document summarizes the performance of GRU models trained on various dataset versions (v2, v3) with 
different feature sets. Each model was tuned using Optuna to find the best hyperparameters.
Window size = 30

* Masking means: For each window feed a mask to the model specifying the valid values with 1 and invalid/missing values with 0

| Model | Best Mean R² | Max R² | Test MSE | Test MAE | Dataset             | Features                        | Masking    | Pad | Criterion method | Model File                    | Hyperparameters                                                                                                                                             |
|-------|--------------|--------|----------|----------|---------------------|---------------------------------|------------|-----|------------------|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | 0.4415       |        | 0.0420   |          | v2                  | VIs(-vari,-gli) + Lepton Temp   | ❌ Disabled | -99 | L1 Loss          | best_model_hyper_param_1.pth  | {'hidden_size': 64, 'num_layers': 1, 'dropout': 0.10344360354230203, 'lr': 0.0002476906474606584, 'weight_decay': 0.0005347879641161282, 'batch_size': 8}   |
| 2     | 0.4605       |        | -        |          | v2                  | VIs(-vari,-gli) + Lepton Temp   | ❌ Disabled | -99 | L1 Loss          | best_model_hyper_param112.pth | {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.03884806018039991, 'lr': 0.00042173321148857924, 'weight_decay': 0.0004972207591747834, 'batch_size': 64} |
| 3     | 0.2857       | 0.6380 | 0.0538   |          | v3                  | Only VIs                        | ✔️ Enabled | 0   | L1  Loss         | best_model_hyper_param_3.pth  | {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.07420456457400194, 'lr': 0.00981026753766152, 'weight_decay': 8.506150589531528e-08, 'batch_size': 16}    |
| 4     | 0.4455       | 0.6007 | 0.0417   |          | v3                  | All VIs + Lepton Temp           | ❌ Disabled | -99 | L1 Loss          | best_model_hyper_param_4.pth  | {'hidden_size': 16, 'num_layers': 1, 'dropout': 0.07545645234219177, 'lr': 0.00038890561009715475, 'weight_decay': 0.0007797370352742492, 'batch_size': 16} |
| 5     | 0.18         |        | -        |          | v3                  | Only VIs                        | ✔️ Enabled | -99 | L1 Loss          | -                             | {'hidden_size': 8, 'num_layers': 2, 'dropout': 0.01882176215855477, 'lr': 0.007580682214882081, 'weight_decay': 1.416934618280146e-06, 'batch_size': 64}    |
| 6     | 0.2585       |        | -        |          | v3                  | Only VIs                        | ✔️ Enabled | 0   | L1 Loss          | -                             | {'hidden_size': 8, 'num_layers': 1, 'dropout': 0.47771061284348876, 'lr': 0.007432078216397578, 'weight_decay': 2.757922254289341e-07, 'batch_size': 32}    |
| 7     | 0.5396       | 0.7620 | 0.0347   | 0.1554   | v3                  | Only VIs                        | ❌ Disabled | 0   | L1 Loss          | best_model_hyper_param_5.pth  | {'hidden_size': 8, 'num_layers': 2, 'dropout': 0.4116615584951404, 'lr': 0.00011452360758206809, 'weight_decay': 2.955054638657815e-08, 'batch_size': 8}    |
| 8     | 0.4071       | 0.6247 | 0.0446   |          | v3                  | All VIs + Lepton Temp           | ❌ Disabled | 0   | L1 Loss          | best_model_hyper_param29.pth  | {'hidden_size': 8, 'num_layers': 3, 'dropout': 0.45863818572312726, 'lr': 0.0001015529310602788, 'weight_decay': 9.401388985791902e-10, 'batch_size': 16}   |
| 9     | 0.3047       | 0.6868 | 0.0523   |          | v3                  | All VIs + Lepton Temp           | ✔️ Enabled | 0   | L1 Loss          | best_model_hyper_param137.pth | {'hidden_size': 8, 'num_layers': 3, 'dropout': 0.49694767108633797, 'lr': 0.00019511281768797167, 'weight_decay': 0.0006741920865352358, 'batch_size': 8}   |
| 10    | 0.5226       | 0.7393 | 0.0359   | 0.1500   | v3_all_interpolated | All VIs + Lepton Temp           | ❌ Disabled | 0   | L1 Loss          | best_model_hyper_param_6.pth  | {'hidden_size': 8, 'num_layers': 3, 'dropout': 0.2870976201082699, 'lr': 0.00010064377208610434, 'weight_decay': 0.0002597190501496971, 'batch_size': 16}   |
| 11    | 0.2667       | 0.7329 | 0.0552   |          | v3_all_interpolated | All VIs + Lepton Temp           | ✔️ Enabled | 0   | L1 Loss          | best_model_hyper_param85.pth  | {'hidden_size': 8, 'num_layers': 3, 'dropout': 0.23655266539255793, 'lr': 0.006284612922199666, 'weight_decay': 1.4279705990700703e-10, 'batch_size': 16}   |
| 12    | 0.3246       | 0.5591 | 0.0509   |          | v3                  | Only VIs and filtered days 1-40 | ❌ Disabled | 0   | L1 Loss          | -                             | {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.050690768780370606, 'lr': 0.007547140411083422, 'weight_decay': 1.2924377235786942e-07, 'batch_size': 8}  |


Window size = 40

| Model | Best Mean R² | Max R² | Test MSE | Test MAE | Dataset | Features | Masking    | Pad | Criterion method | Model File | Hyperparameters                                                                                                                                          |
|-------|--------------|--------|----------|----------|---------|----------|------------|-----|------------------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| 13    | 0.4961       | 0.6857 | 0.0379   | 0.1572   | v3      | Only VIs | ❌ Disabled | 0   | L1 Loss          |            | {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.14212952905983112, 'lr': 0.006344969970535884, 'weight_decay': 3.119218313061341e-05, 'batch_size': 8} |           


Window size = 20

| Model | Best Mean R² | Max R² | Test MSE | Test MAE | Dataset             | Features              | Masking    | Pad | Criterion method | Model File                   | Hyperparameters                                                                                                                                             |
|-------|--------------|--------|----------|----------|---------------------|-----------------------|------------|-----|------------------|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 14    | 0.4274       | 0.7529 | 0.0431   | 0.1660   | v3                  | Only VIs              | ❌ Disabled | 0   | L1 Loss          | best_model_hyper_param_7.pth | {'hidden_size': 8, 'num_layers': 3, 'dropout': 0.15407524085698027, 'lr': 0.00022516408745938045, 'weight_decay': 0.0001017303692011807, 'batch_size': 64}  |           
| 15    | 0.3467       | 0.7493 | 0.0492   | 0.1762   | v3_all_interpolated | All VIs + Lepton Temp | ❌ Disabled | 0   | L1 Loss          | best_model_hyper_param_8.pth | {'hidden_size': 8, 'num_layers': 3, 'dropout': 0.29994133954772173, 'lr': 0.00022398230500393924, 'weight_decay': 0.00015160363384374365, 'batch_size': 64} |


Window size = 25

| Model | Best Mean R² | Max R² | Test MSE | Test MAE | Dataset | Features | Masking    | Pad | Criterion method | Model File                   | Hyperparameters                                                                                                                                            |
|-------|--------------|--------|----------|----------|---------|----------|------------|-----|------------------|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 16    | 0.5003       | 0.7332 | 0.0376   | 0.1589   | v3      | Only VIs | ❌ Disabled | 0   | L1 Loss          | best_model_hyper_param_9.pth | {'hidden_size': 8, 'num_layers': 3, 'dropout': 0.10503772376300814, 'lr': 0.00011690368125953273, 'weight_decay': 0.0009878712546328788, 'batch_size': 32} |           

Window size = 35

| Model | Best Mean R² | Max R² | Test MSE | Test MAE | Dataset | Features | Masking    | Pad | Criterion method | Model File                    | Hyperparameters                                                                                                                                             |
|-------|--------------|--------|----------|----------|---------|----------|------------|-----|------------------|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 17    | 0.5235       | 0.6951 | 0.0359   | 0.1540   | v3      | Only VIs | ❌ Disabled | 0   | L1 Loss          | best_model_hyper_param_10.pth | {'hidden_size': 8, 'num_layers': 2, 'dropout': 0.15733282331703996, 'lr': 0.00048561958669596813, 'weight_decay': 0.00018153262022945748, 'batch_size': 32} |           


### LSTM Model Hyperparameter Tuning Results

This document summarizes the performance of LSTM models trained on v3 dataset with 
different feature sets. Each model was tuned using Optuna to find the best hyperparameters.
Window size = 30

| Model | Best Mean R² | Max R² | Test MSE | Test MAE | Dataset             | Features              | Masking    | Pad | Criterion method | Model File                   | Hyperparameters                                                                                                                                           |
|-------|--------------|--------|----------|----------|---------------------|-----------------------|------------|-----|------------------|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| 16    | 0.3636       | 0.5174 | 0.0479   | 0.1732   | v3                  | Only VIs              | ❌ Disabled | 0   | L1 Loss          | best_model_hyper_param_2.pth | {'hidden_size': 8, 'num_layers': 3, 'dropout': 0.27260383547219424, 'lr': 0.009285505811941576, 'weight_decay': 3.165581570390733e-07, 'batch_size': 8}   |          
| 17    | 0.2652       | 0.4955 | 0.0553   | 0.1891   | v3_all_interpolated | All VIs + Lepton Temp | ❌ Disabled | 0   | L1 Loss          | best_model_hyper_param_3.pth | {'hidden_size': 16, 'num_layers': 2, 'dropout': 0.4248770561325045, 'lr': 0.0033221301193583374, 'weight_decay': 2.3506292058834832e-08, 'batch_size': 8} |
