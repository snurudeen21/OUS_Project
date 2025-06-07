import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import fixed_quad
import matplotlib.pyplot as plt

# File path (Update accordingly)
file_path = 'C:\\Users\\user\\OneDrive - UGent\\EmShip\\OUS\\Prototip.xls'

if file_path:
    # Load the Excel file
    excel_data = pd.ExcelFile(file_path)

    # Parse the required sheet
    sheet_data = excel_data.parse('Sheet7')

    # Clean the data by removing unnecessary columns and dropping NaN values
    cleaned_data = sheet_data.drop(columns=['Unnamed: 8', 'Unnamed: 9'], errors='ignore').dropna()

    # Convert the cleaned DataFrame to a list of lists (excluding headers)
    data_as_list = cleaned_data.values.tolist()

    ship_data = []
    for i in range(len(cleaned_data.columns)):
        column_data = [row[i] for row in data_as_list]
        ship_data.append(column_data)

print(ship_data[6])  # Debugging output
print("\n")

# Extract x-values (integration variable)
x = np.array(ship_data[1])

# Compute function values to integrate
integral = np.array([w * y * z for w, y, z in zip(ship_data[3], ship_data[4], ship_data[4])])
integral_1 = np.array([a * b * c for a, b, c in zip(ship_data[3], ship_data[4], ship_data[6])])

# Define interpolating function for Gaussian Quadrature
interp_func = PchipInterpolator(x, integral)
interp_func_1 = PchipInterpolator(x, integral_1)

# Apply Gaussian Quadrature with 5 integration points (can increase for higher accuracy)
result, error = fixed_quad(interp_func, x.min(), x.max(), n=5)
result_1, error_1 = fixed_quad(interp_func_1, x.min(), x.max(), n=5)

print(f"Integral (Gaussian Quadrature): {result}\n")
print(f"Integral_1 (Gaussian Quadrature): {result_1}")

# Plotting section remains unchanged
t = np.linspace(0, 60, 120)
#val = [-3.79E+03, 4.50E+01, 0.390259957]    #sheet 6
val = [-3.79E+03, 3.36E+03, 0.390259957]     #sheet 7

f_t = [val[0] * np.cos(val[2] * k) + val[1] * np.sin(val[2] * k) for k in t]

plt.figure(figsize=(10, 6))
plt.scatter(t, f_t, color='red', label='Original data', zorder=3)
plt.plot(t, f_t, label='curve')

plt.xlabel('t')
plt.ylabel('f_t')
plt.legend()
plt.grid(True)
plt.show()