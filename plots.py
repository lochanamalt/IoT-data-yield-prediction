import numpy as np
from matplotlib import pyplot as plt
import datetime as dt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

START_DAY = 122 # May 1st in gregorian days


def plot_r2(r2_list, total_n_days):
    x = np.arange(START_DAY, START_DAY + total_n_days)
    max_idx = np.argmax(r2_list)
    max_x = x[max_idx]
    max_y = r2_list[max_idx]

    r2_copy = np.delete(r2_list, max_idx)  # remove max
    second_max_val = np.max(r2_copy)
    second_max_idx = np.where(r2_list == second_max_val)[0][0]
    second_max_x = x[second_max_idx]

    plt.figure(figsize=(12, 4))
    # plt.axvline(x=143, color='gray', linestyle='--', linewidth=1,
    #             label='Heading date')
    plt.axvspan(133,152,color='gray',alpha=0.3,label='Heading window')


    plt.xlabel("Day (Gregorian days)")
    plt.plot(x, r2_list, marker='o')
    plt.scatter(max_x, max_y, color='red', s=80, zorder=5)
    plt.text(max_x, max_y, f" {max_y:.4f} at day {max_x}", color='red', fontsize=10, va='bottom')
    # plt.scatter(second_max_x, second_max_val, color='red', s=80, zorder=5)
    # plt.text(second_max_x, second_max_val, f"  {second_max_val:.4f} at day {second_max_x}", color='red',
    #          fontsize=10, va='bottom')
    plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)


    plt.ylabel("Test R²")
    # plt.title("Test set drought plots R² over days")
    tick_indices = np.arange(0, len(x), 2)
    tick_values = x[tick_indices]

    plt.xticks(ticks=tick_values, ha='right')

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

    # plt.scatter(second_min_x, second_min_val, color='red', s=80, zorder=5, label="Second Min Value")
    # plt.text(second_min_x, second_min_val, f"  min={second_min_y_rounded} at day {second_min_x}", color='red', fontsize=10, va='bottom')



    plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
    plt.xlabel("Day (Gregorian days)")
    plt.ylabel("Test MSE")
    plt.title("Test MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_predicted_yield(total_n_days: int, y_pred_list, actual_yield: float, heading_date_for_plot: int, plot_name: str):
    x = np.arange(START_DAY, START_DAY + total_n_days)
    date_labels = [(dt.date(2024, 1, 1) + dt.timedelta(days=int(day) - 1)).strftime("%b %d")
                   for day in x]

    differences = np.abs(np.array(y_pred_list) - actual_yield)
    closest_idx = np.argmin(differences)

    closest_day = x[closest_idx]
    closest_pred = y_pred_list[closest_idx]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y_pred_list, marker='o', label='Predicted Yield')

    plt.axhline(actual_yield, linestyle='--', label=f'Actual Yield ({actual_yield:.2f} kg/m²)', color='r')
    # plt.scatter(closest_day, closest_pred, s=120, edgecolors='black',
    #             label=f'Closest Predicted Yield ({closest_pred:.2f} kg/m²) at day {closest_day}', zorder=5)

    plt.axvline(x=heading_date_for_plot, color='gray', linestyle='--', linewidth=1,
                label='Heading date')

    tick_indices = np.arange(0, len(x), 2)  # every 6th point
    tick_values = x[tick_indices]
    tick_labels = [date_labels[i] for i in tick_indices]

    plt.xlabel("Date")
    plt.ylabel("Predicted yield (kg/m²)")
    plt.xticks(ticks=tick_values, labels=tick_labels, rotation=45, ha='right')

    plt.title(f"Predicted yield over days vs actual yield for plot {plot_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_r2_with_another_axis(r2_list, axis_2_data, axis_2_name, total_n_days, output_filename):
    x = np.arange(START_DAY, START_DAY + total_n_days)
    max_idx = np.argmax(r2_list)
    max_x = x[max_idx]
    max_y = r2_list[max_idx]

    fig, ax1 = plt.subplots(figsize=(12, 4))


    ax1.set_xlabel("Day (Gregorian days)")
    ax1.plot(x, r2_list, marker='o')
    ax1.set_ylabel("Test R²")

    ax1.scatter(max_x, max_y, color='red', s=80, zorder=5)

    ax2 = ax1.twinx()  # second y-axis
    ax2.plot(x, axis_2_data, color='green', label=axis_2_data)
    ax2.set_ylabel(axis_2_name)




    ax1.text(max_x, max_y, f" {max_y:.4f} at day {max_x}", color='red', fontsize=10, va='bottom')
    # plt.scatter(second_max_x, second_max_val, color='red', s=80, zorder=5)
    # plt.text(second_max_x, second_max_val, f"  {second_max_val:.4f} at day {second_max_x}", color='red',
    #          fontsize=10, va='bottom')

    ax1.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)


    # plt.title("Test R² over days")
    tick_indices = np.arange(0, len(x), 2)
    tick_values = x[tick_indices]

    plt.axvspan(133,152,color='gray',alpha=0.3,label='Heading window')
    plt.xticks(ticks=tick_values, ha='right')
    plt.grid(True)
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Test R²'),
        Line2D([0], [0], color='green', lw=2, label=axis_2_name),
        Patch(facecolor='gray', alpha=0.3, label='Heading window')
    ]

    ax1.legend(handles=legend_elements, loc='best')

    # plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
