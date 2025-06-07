import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simpson, trapezoid
import matplotlib.pyplot as plt
import pandas as pd

'''# Original data
z = np.array([1.86666666666667,3.73333333333333,5.6,7.46666666666667, 8.4])
#z = np.array([1.86666666666667,3.73333333333333,5.6,7.46666666666667, 8.4])
y=np.array([1.29926946107784,1.68582035928144,1.73741748572825,1.8817005988024,2.08904191616766])

#y=np.array([-72.975,-69.5,-66.025,-62.55,-59.075,-55.6,-52.125,-48.65,-45.175,-41.7,-38.225,-34.75,-31.275,-27.8,-24.325,-20.85,-17.375,-13.9,-10.425,-6.95,-3.475,0,3.475,6.95,10.425,13.9,17.375,20.85,24.325,27.8,31.275,34.75,38.225,41.7,45.175,48.65,52.125,55.6,59.075,62.55,66.025,69.5,72.975,76.45])



# Create PCHIP interpolator
interpolator = PchipInterpolator(z, y)

# Generate new points for integration (more points = more accurate)
z_new = np.linspace(z.min(), z.max(), 1000)  # From first to last z point
y_new = interpolator(z_new)

# Calculate area using Simpson's rule
area = simpson(y_new, z_new)
area_1 = trapezoid(y_new, z_new)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y, z, color='red', label='Original data', zorder=3)
plt.plot(y_new, z_new, label='PCHIP interpolation')
plt.fill_between(y_new, z_new, alpha=0.2, label='Area to integrate')
plt.title(f'PCHIP Interpolation and Integration\nArea = {area:.4f}')
plt.xlabel('y')
plt.ylabel('z')
plt.legend()
plt.grid(True)
plt.show()

print(f"Calculated area using Simpson's rule: {area:.6f}")
print(f"Calculated area using Trapezoid's rule: {area_1:.6f}")'''


#roots from a graph
from scipy.interpolate import PchipInterpolator
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

def equation(k, k0, h):
    return k0 / k - np.tanh(k * h)

# Given parameters
k0 = 0.015525263
h = 131

# Define the function for root finding
def f(k):
    return equation(k, k0, h)

# Find the root using Brent's method (good for bracketed roots)
# We need to find a reasonable interval where the root exists
# Let's first plot to estimate the root location (optional)

k_values = np.linspace(0.0001, 0.1, 1000)
f_values = [f(k) for k in k_values]

plt.plot(k_values, f_values)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('k')
plt.ylabel('f(k)')
plt.title('Function to find root')
plt.grid()
plt.show()

# From the plot, we can see the root is around 0.0001 to 0.001
# Let's find it precisely
solution = root_scalar(f, bracket=[0.01, 0.02], method='brentq')

print(f"The solution is k = {solution.root:.10f}")
print(f"Verification: f({solution.root:.10f}) = {f(solution.root):.10e}")







#get a value along a piecewise function
x = np.array([0,0.0125,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1])
#z = np.array([0,1.86666666666667,3.73333333333333,5.6,7.46666666666667,9.33333333333333,11.2,13.0666666666667,14.9333333333333,16.8])

y=np.array([0,0.03315,0.04576,0.06221,0.0735,0.08195,0.09354,0.1004,0.10397,0.10504,0.10156,0.09265,0.07986,0.06412,0.04591,0.02534,0.01412,0.00221])

interpolator = PchipInterpolator(x, y)
x_new = np.linspace(0, 1, 25)
y_new = interpolator(x_new)

combine = []
combine.append(x_new)
combine.append(y_new)
#x_target = 8.4  # The x value you want the corresponding y for
#y_target = interpolator(x_target)  # Interpolated y value

#print(f"The interpolated y value at z = {x_target} is {y_target}")

df = pd.DataFrame(combine)

# Save to Excel in a single row
output_file = 'C:\\Users\\user\\Desktop\\Areas_Output.xlsx'
df.to_excel(output_file, index=False, header=False)

#plt.plot(x, y, 'o', label="Data points")
plt.plot(x_new, y_new, label="PCHIP Interpolation")
plt.legend()
plt.show()