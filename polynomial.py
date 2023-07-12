# Example: Predict price, given pizza crust thickness.
# Non-Linear Data (Regression Problem)
#
# If data is non-linear, you can use a polynomial as predictor.
# Task: "Predict price, given non-linear data"
#


import numpy as np
import matplotlib.pyplot as plt

# Pizza crust thickness
X = [1, 5, 8, 10, 14, 18]
# Price
Y = [1, 1, 5, 6, 15, 25]

# Train Algorithm (Polynomial)
degree = 2
poly_fit = np.poly1d(np.polyfit(X,Y, degree))

# Create plot to show data and predictions
xx = np.linspace(0, 26, 100)
plt.plot(xx, poly_fit(xx), c='r',linestyle='-')
plt.title('Pizza price regressed on height')
plt.xlabel('Height in mm')
plt.ylabel('Price $')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X, Y)
plt.show()

# Predict price, given thickness of 10mm
print( poly_fit(10) )
