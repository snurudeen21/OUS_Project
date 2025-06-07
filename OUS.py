import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simpson, trapezoid
import matplotlib.pyplot as plt


k = 0.016001622
a = np.cos(k*3)
print(f"{a}\n")

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

z = np.array([0,1.86666666666667, 3.73333333333333, 5.6, 7.46666666666667,8.4])
z_new = np.linspace(z.min(), z.max(), 1000)

areas = []

for i in range(len(ship_data)):
    y = np.array(ship_data[i])
    
    interpolator = PchipInterpolator(z, y)

    # Generate new points for integration (more points = more accurate)
      # From first to last z point
    y_new = interpolator(z_new)
    area = simpson(y_new, z_new)
    areas.append(area)


areas = [y*2 for y in areas]


df = pd.DataFrame([areas])

# Save to Excel in a single row
output_file = 'C:\\Users\\user\\Desktop\\Areas_Output.xlsx'
df.to_excel(output_file, index=False, header=False)

x_val = np.array([0,1.86666666666667, 3.73333333333333, 5.6, 7.46666666666667,8.4])
z_new = np.linspace(z.min(), z.max(), 1000)