"""
Plot the number of samples over epochs
"""

from matplotlib import pyplot as plt
import numpy as np

dens_interval = 100

a = [120, 35, 27, 23, 32, 46, 49, 46, 54, 55, 54, 53, 54, 55, 58, 61, 
     56, 53, 58, 62, 51, 70, 64, 55, 47, 58, 56, 59, 67, 64, 62, 48, 69, 
     60, 72, 60, 66, 70, 59, 59, 50, 56, 54, 71, 53, 57, 55, 53, 56]

x = 0
b = []
for i in range(len(a)): 
    x = a[i] + x
    b.append(x)

print(b)

c = []

for i in range(len(b)):
    for j in range(dens_interval):
        c.append(b[i])
    
plt.plot(range(len(c)), c)
plt.title('Number of Samples vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Number of Samples')
file_path = 'Number_of_samples_100.png'
plt.savefig(file_path, bbox_inches='tight')
plt.clf()  # Clear the current figure
plt.close()  # Close the current figure
