import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simpson, trapezoid
import matplotlib.pyplot as plt


file_path = 'C:\\Users\\user\\OneDrive - UGent\\EmShip\\OUS\\Prototip.xls'
#file_path = ''


if (file_path):
    # Load the Excel file
    excel_data = pd.ExcelFile(file_path)

    # Parse the required sheet
    sheet_data = excel_data.parse('Sheet6')


    # Clean the data by removing unnecessary columns and dropping rows with NaN values
    cleaned_data = sheet_data.drop(columns=['Unnamed: 8', 'Unnamed: 9'], errors='ignore').dropna()

    # Extract column headers as a list
    '''headers = cleaned_data.columns.tolist()

    y_values = []

    for i in range(0,len(headers)):
        y_values.append(headers[i])'''
    
    # Convert the cleaned DataFrame to a list of lists (excluding the headers) in rows
    data_as_list = cleaned_data.values.tolist()

    ship_data = []
   
    for i in range(len(cleaned_data.columns)):
        column_data = [row[i] for row in data_as_list]
        ship_data.append(column_data)

print(ship_data[6])
print("\n")
x = np.array(ship_data[1])

x_new = np.linspace(x.min(), x.max(), 1000)

integral = [w * y * z for w, y, z in zip(ship_data[3], ship_data[4], ship_data[4])]
integral_1 = [a * b * c for a, b, c in zip(ship_data[3], ship_data[4], ship_data[6])]

y = np.array(integral)
y_1 = np.array(integral_1)



# Generate new points for integration (more points = more accurate)
# From first to last z point


result = simpson(y,x)
result_1 = simpson(y_1,x)

print(f"{result}\n")
print(f"{result_1}")

t = np.linspace(0,60,120)
val = [-3.79E+03, 4.50E+01, 0.390259957]    #sheet 6
#val = [-3.79E+03, 3.36E+03, 0.390259957]     #sheet 7

f_t = [val[0]*np.cos(val[2]*k) + val[1]*np.sin(val[2]*k) for k in t]

plt.figure(figsize=(10, 6))
plt.scatter(t, f_t, color='red', label='Data Points', zorder=3)
plt.plot(t, f_t, label='curve')


plt.xlabel('t (s)')
plt.ylabel('F (kN)')
plt.legend()
plt.grid(True)
plt.show()