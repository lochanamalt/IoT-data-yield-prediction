import numpy as np
from matplotlib import pyplot as plt

START_DAY = 122 # May 1st in gregorian days

def plot_r2(r2_list, total_n_days):
    x = np.arange(START_DAY, START_DAY + total_n_days)
    max_idx = np.argmax(r2_list)
    max_x = x[max_idx]
    max_y = r2_list[max_idx]
    max_y_rounded = round(max_y, 2)
    r2_copy = np.delete(r2_list, max_idx)  # remove max
    second_max_val = np.max(r2_copy)
    second_max_idx = np.where(r2_list == second_max_val)[0][0]
    second_max_x = x[second_max_idx]
    second_max_y_rounded = round(second_max_val, 2)
    plt.figure(figsize=(12, 4))
    plt.axvline(x=143, color='gray', linestyle='--', linewidth=1,
                label='Heading date')
    plt.plot(x, r2_list, marker='o')
    plt.scatter(max_x, max_y, color='red', s=80, zorder=5, label="Max Value")
    plt.text(max_x, max_y, f"  max={max_y_rounded} at day {max_x}", color='red', fontsize=10, va='bottom')
    plt.scatter(second_max_x, second_max_val, color='red', s=80, zorder=5, label="Second Max Value")
    plt.text(second_max_x, second_max_val, f"  max={second_max_y_rounded} at day {second_max_x}", color='red',
             fontsize=10, va='bottom')
    plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
    plt.xlabel("Upto Days")
    plt.ylabel("Test R²")
    plt.title("Test R² over days")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mse(mse_list, total_n_days):
    x = np.arange(START_DAY, START_DAY + total_n_days)
    min_idx = np.argmin(mse_list)
    min_x = x[min_idx]
    min_y = mse_list[min_idx]
    min_y_rounded = round(min_y, 2)

    test_mse_copy = np.delete(mse_list, min_idx)      # remove max
    second_min_val = np.min(test_mse_copy)
    second_min_idx = np.where(mse_list == second_min_val)[0][0]
    second_min_x = x[second_min_idx]
    second_min_y_rounded = round(second_min_val, 2)

    plt.figure(figsize=(12,4))
    plt.plot(x, mse_list, marker='o')
    plt.axvline(x=143, color='gray', linestyle='--', linewidth=1,
                label='Heading date')

    plt.scatter(min_x, min_y, color='red', s=80, zorder=5, label="Min Value")
    plt.text(min_x, min_y, f"  min={min_y_rounded} at day {min_x}", color='red', fontsize=10, va='bottom')

    plt.scatter(second_min_x, second_min_val, color='red', s=80, zorder=5, label="Second Min Value")
    plt.text(second_min_x, second_min_val, f"  min={second_min_y_rounded} at day {second_min_x}", color='red', fontsize=10, va='bottom')



    plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
    plt.xlabel("Day (Gregorian days)")
    plt.ylabel("Test MSE")
    plt.title("Test MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_predicted_yield(total_n_days: int, y_pred_list, actual_yield: float):
    x = np.arange(START_DAY, START_DAY + total_n_days)

    differences = np.abs(np.array(y_pred_list) - actual_yield)
    closest_idx = np.argmin(differences)

    closest_day = x[closest_idx]
    closest_pred = y_pred_list[closest_idx]


    plt.figure(figsize=(8, 4))
    plt.plot(x, y_pred_list, marker='o', label='Predicted Yield')

    plt.axhline(actual_yield, linestyle='--', label=f'Actual Yield ({actual_yield:.2f} kg/m²)', color='r')
    plt.scatter(closest_day, closest_pred, s=120, edgecolors='black',
                label='Closest Predicted Yield', zorder=5)

    tick_indices = np.arange(0, len(x), 2)  # every 6th point
    tick_values = x[tick_indices]
    tick_labels = [date_labels[i] for i in tick_indices]

    plt.xlabel("Day (Gregorian days)")
    plt.ylabel("Predicted yield (kg/m²)")

    plt.title("Predicted Yield Over Days vs Actual Yield")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()