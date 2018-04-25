# import numpy as np
# import matplotlib.pyplot as plt

mu, sigma = 0, 10
dataSigma = 7
noiseSigma = 10

n = 5
centroidsCount = 5
noiseCount = 30
centroids = []
xs = []
ys = []
for i in range(0, centroidsCount):
    xi = np.random.normal(mu, sigma)
    yi = np.random.normal(mu, sigma)
    centroids.append((xi, yi))
    xs.append(xi)
    ys.append(yi)

points = []
dataXs = []
dataYs = []
for element in centroids:
    for i in range(0, n):
        xn = np.random.normal(element[0], dataSigma)
        yn = np.random.normal(element[1], dataSigma)
        points.append((xn, yn))
        dataXs.append(xn)
        dataYs.append(yn)

noiseXs = []
noiseYs = []
for i in range(0, noiseCount):
    xi = np.random.normal(mu, noiseSigma)
    yi = np.random.normal(mu, noiseSigma)
    points.append((xi, yi))
    noiseXs.append(xi)
    noiseYs.append(yi)

plt.plot(dataXs, dataYs, 'ro')
plt.plot(noiseXs, noiseYs, 'yo')
plt.plot(xs, ys, 'bo')
plt.show()
