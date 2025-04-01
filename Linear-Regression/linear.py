import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initialize the Linear Regression model.

        Parameters:
        lr (float): Learning rate for gradient descent. Default is 0.001.
        n_iters (int): Number of iterations for gradient descent. Default is 1000.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters:
        X (numpy array): Training data, shape (n_samples, n_features).
        y (numpy array): Target values, shape (n_samples,).

        Updates:
        self.weights (numpy array): Coefficients for each feature.
        self.bias (float): Intercept of the model.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # Predicted values
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradient of cost function w.r.t. weights and bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # Update weights and bias using gradient descent
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict target values for new data.

        Parameters:
        X (numpy array): New data to predict, shape (n_samples, n_features).

        Returns:
        y_pred (numpy array): Predicted target values, shape (n_samples,).
        """
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
