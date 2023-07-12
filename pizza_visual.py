# This program makes a plot.
#
# Plot with data: 
#  + diameter of pizza (X, horizontal axis)
#  + price in dollars (Y, vertical axis)
# 

# Use Matplotlib 
import matplotlib.pyplot as plt

# diameter of pizza
X = [[4], [8], [12], [16], [18]]
# price
y = [[4], [8], [10], [12], [15]]

# Create plot with matplotlib
plt.figure()
plt.title('Pizza price vs diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, y, marker="*",color='red')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()
