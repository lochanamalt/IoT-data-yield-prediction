"""
@author: Lochana Marasinghe
@date: 1/14/2026
@description: 
"""
import pandas as pd

from plots import plot_r2_with_another_axis

df = pd.read_csv("results/best_gru_model_14_seed_46_performance_with_gridmet.csv")

test_r2 = df["R2"].to_list()
test_mse = df["MSE"].to_list()
test_mae = df["MAE"].to_list()
# solar = df["Solar"].to_list()
# precipitation = df["Precipitation"].to_list()
# air_temp = df["AirTemp"].to_list()
# vapor_pressure = df["Vapor_Pressure"].to_list()
# atm_pressure = df["AtmPressure"].to_list()
# rel_humidity = df["RelHumidity"].to_list()
# rel_humidity_percentage = [x * 100 for x in rel_humidity]
# wind_speed = df["WindSpeed"].to_list()


precipitation = df["pr"].to_list()
air_temp = df["tavg"].to_list()
gdd = df["gdd"].to_list()
cgdd = df["cgdd"].to_list()


total_days = 46

plot_r2_with_another_axis(r2_list=test_r2,axis_2_data=air_temp,axis_2_name="Air Temperature (°C)",
                          total_n_days=total_days, output_filename="results/model_14_seed_46_r2_with_air_temp_gridmet.png")
# plot_r2_with_another_axis(r2_list=test_r2,axis_2_data=precipitation,axis_2_name="Precipitation (mm)",
#                           total_n_days=total_days, output_filename="results/r2_with_precipitation_gridmet.png")
# plot_r2_with_another_axis(r2_list=test_r2,axis_2_data=gdd,axis_2_name="Growing Degree Days",
#                           total_n_days=total_days, output_filename="results/r2_with_gdd_gridmet.png")
# plot_r2_with_another_axis(r2_list=test_r2,axis_2_data=cgdd,axis_2_name="Accumulated Growing Degree Days",
#                           total_n_days=total_days, output_filename="results/r2_with_cgdd_gridmet.png")
# plot_r2_with_another_axis(r2_list=test_r2,axis_2_data=atm_pressure,axis_2_name="Atmospheric Pressure (kPa)",
#                           total_n_days=total_days, output_filename="results/r2_with_atm_pressure.png")
# plot_r2_with_another_axis(r2_list=test_r2,axis_2_data=rel_humidity_percentage,axis_2_name="Relative Humidity (%)",
#                           total_n_days=total_days, output_filename="results/r2_with_rel_humid.png")
# plot_r2_with_another_axis(r2_list=test_r2,axis_2_data=solar,axis_2_name="Solar Radiation (W/m²)",
#                           total_n_days=total_days, output_filename="results/r2_with_solar_radiation.png")

