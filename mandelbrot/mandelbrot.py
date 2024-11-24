import numpy as np
import matplotlib.pyplot as plt


size = 251 # must be odd, size of image
size3_4 = (int)(((size - 1) * 3 / 4) + 1)
size4_5 = (int)(((size - 1) * 3 / 5) + 1)
#points = [[0 for i in range(size)] for j in range(size)]
values = [[0 for i in range(size3_4)] for j in range(size4_5)]

step = 4 / ((size) - 1)
"""
for i in range(size):
    for j in range(size):
        points[j][i] = np.complex(-2 + (step * i), -2 + (step * j))
"""

maxiterations = 200 # increase for better image

# i made thsi years ago and i don't remember why most of this is the way it is
def mandel(z, x, y, iteration):
    c = complex(-2 + (step * x), -1.2 + (step * y))
    z_2 = np.power(z, 2)
    if np.absolute(z) < 100:
        if iteration < maxiterations:
            mandel(z_2 + c, x, y, iteration + 1)
        else:
            values[y][x] = 0
    else:
        values[y][x] = iteration


for i in range(size3_4):
    for j in range(size4_5):
        print(str(i) + ", " + str(j))
        mandel(0, i, j, 0)
#print(points)



plt.figure(figsize=(8, 8))
plt.imshow(values, cmap = 'CMRmap', interpolation = 'none')
#plt.savefig('mandelbrot_6.pdf', bbox_inches='tight')
plt.show()