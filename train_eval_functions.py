from typing import Tuple

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader
from model_definitions import GRUModel, LSTMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_data(model, model_train_loader, num_epochs, learning_rate, weight_decay, hyper_param_criterion_method,
               file_save_name, early_stopping = False):
    best_train_loss  = float('inf')
    patience = 20
    counter = 0

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    if hyper_param_criterion_method == "MSE":
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss(beta=1)

    total_loss = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch, lengths in model_train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch, lengths)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        avg_loss = train_loss / len(model_train_loader.dataset)
        total_loss.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.5f}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            if file_save_name != "":
                torch.save(model.state_dict(), file_save_name)
            counter = 0
        elif early_stopping:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
    return total_loss



def evaluate_model(model, val_loader, output_variable_scaler, do_inverse_transform = False,
                   plot_pred_vs_true = False) -> Tuple[float, float, float]:

    model.eval()  # set model to evaluation mode
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch, lengths in val_loader:
            X_batch = X_batch.to(device)
            lengths = lengths.to(device)
            y_hat = model(X_batch, lengths)

            y_true.append(y_batch.cpu())
            y_pred.append(y_hat.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    if do_inverse_transform:
        y_true = output_variable_scaler.inverse_transform(y_true)
        y_pred = output_variable_scaler.inverse_transform(y_pred)

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    if plot_pred_vs_true:
        plt.figure(figsize=(8,6))
        plt.scatter(y_true, y_pred, alpha=0.7)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
        plt.xlabel("Actual Yield")
        plt.ylabel("Predicted Yield")
        plt.title("LSTM Predictions vs Actual")
        plt.show()

    return r2, mse, mae


def validation_with_test(trial_id, train_dataset, test_dataset,features, scaler_y, batch_size, num_epochs, hidden_size, num_layers,
                         dropout, bidirectional, learning_rate, weight_decay, hyper_param_criterion_method):

    train_loader_tuning  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader_tuning    = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GRUModel(input_size = len(features), hidden_size = hidden_size, num_layers = num_layers,
                     output_size= 1, dropout=dropout, bidirectional = bidirectional).to(device)

    train_data(
        model = model,
        model_train_loader= train_loader_tuning,
        num_epochs = num_epochs,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        hyper_param_criterion_method = hyper_param_criterion_method,
        file_save_name=f"best_model_hyper_param{trial_id}.pth",
        early_stopping=True
    )

    r2, mse, mae = evaluate_model(
        model = model,
        val_loader = test_loader_tuning,
        output_variable_scaler=scaler_y)

    print(f" R²: {r2:.4f}")
    return r2


def validation_with_test_LSTM(trial_id, train_dataset, test_dataset, features, scaler_y, batch_size, num_epochs, hidden_size, num_layers,
                              dropout, learning_rate, weight_decay, hyper_param_criterion_method):

    train_loader_tuning  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader_tuning    = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size = len(features), hidden_size = hidden_size, num_layers = num_layers,
                     output_size= 1, dropout=dropout).to(device)

    train_data(
        model = model,
        model_train_loader= train_loader_tuning,
        num_epochs = num_epochs,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        hyper_param_criterion_method = hyper_param_criterion_method,
        file_save_name=f"best_model_hyper_param{trial_id}.pth",
        early_stopping=True
    )

    r2, mse, mae = evaluate_model(
        model = model,
        val_loader = test_loader_tuning,
        output_variable_scaler=scaler_y)

    print(f" R²: {r2:.4f}")
    return r2