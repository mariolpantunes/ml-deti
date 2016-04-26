import matplotlib.pyplot as plt
import arff
import numpy as np
from sklearn import linear_model

# Load dataset
dataset = arff.load(open('dataset/dataset00.arff', 'r'))
data = np.array(dataset['data'])

# Reshape vector
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)

# Plot points
plt.scatter(X, Y,  color='black')
plt.xticks(())
plt.yticks(())
plt.show()

# Create linear regression object
model = linear_model.LinearRegression()

# Train the model using X and Y
model.fit(X, Y)

# The coefficients
print("Y = %.2fX + %.2f" % (model.coef_, model.intercept_))
# The mean square error
print("Residual sum of squares: %.2f" % np.mean((model.predict(X) - Y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model.score(X, Y))

# Plot outputs
plt.scatter(X, Y,  color='black')
plt.plot(X, model.predict(X), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())

plt.show()

plt.plot([1,2,3,4], [0,0,0,0], 'ro')
plt.plot([6,7,8,9], [1,1,1,1], 'bs')
plt.axis([0, 10, -1, 2])
plt.show()
