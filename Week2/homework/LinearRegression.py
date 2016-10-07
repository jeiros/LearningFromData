from matplotlib import pyplot as plt
import numpy as np
import random as rd


class LinearRegression(object):

    """docstring for LinearRegression"""

    def __init__(self, N_points):
        self.N_points = N_points
        xA, yA, xB, yB = [rd.uniform(-1, 1) for i in range(4)]
        self.V = np.array([xB * yA - xA * yB, yB - yA, xA - xB])

    def generate_points(self):
        X = np.empty(shape=(0, 4))
        # Generate random points and label them with the previous line
        for i in range(self.N_points):
            x1, x2 = [rd.uniform(-1, 1) for i in range(2)]
            x = np.array([1, x1, x2])
            s = int(np.sign(self.V.T.dot(x)))
            x_labeld = np.hstack([x, s])
            X = np.vstack([X, x_labeld])
        self.X = X
        return(X, self.V)

    def generate_Noisy_points(self):
        X = np.empty(shape=(0, 4))
        for i in range(self.N_points):
            x1, x2 = [rd.uniform(-1, 1) for i in range(2)]
            x = np.array([1, x1, x2])
            s = int(np.sign(x1 * x1 + x2 * x2 - 0.6))
            if rd.random() <= 0.1:
                # 'Flip label with a 10% chance'
                s = s * -1
            x_labeld = np.hstack([x, s])
            X = np.vstack([X, x_labeld])
        self.X = X
        return(X)

    def transformInput(self):
        X_trans = np.empty(shape=(0, 6))
        for point in self.X[:, 0:3]:
            x1x2 = point[1] * point[2]
            x1x1 = point[1] * point[1]
            x2x2 = point[2] * point[2]
            X_trans = np.vstack([X_trans, np.hstack([point, x1x2, x1x1, x2x2])])
        self.X_trans = np.hstack([X_trans, np.reshape(self.X[:, 3], newshape=(1000, 1))])
        return(self.X_trans)

    def applyLine(self, vector, point):
        A = vector[0]
        B = vector[1]
        C = vector[2]
        y = (-B / C) * point + (-A / C)
        return(y)

    def estimate_line(self, feature_mat):
        points = feature_mat[:, :-1]
        labels = feature_mat[:, -1]
        X_dagger = np.linalg.inv(points.T.dot(points)).dot(points.T)
        w = X_dagger.dot(labels)
        self.w = w
        return(w)

    def estimate_Ein(self):
        "The fraction of in-sample points which got classified incorrectly"
        try:
            pred_labels = np.sign(self.w.T.dot(self.X[:, 0:3].T))
            errors = np.count_nonzero(pred_labels != self.X[:, 3])
            return(errors / self.N_points)
        except:
            raise ValueError('Generate points and estimate g(x) first please.')

    def plot(self):
        reds = self.X[self.X[:, 3] == -1]
        blues = self.X[self.X[:, 3] == 1]
        plt.scatter(reds[:, 1], reds[:, 2], color='red')
        plt.scatter(blues[:, 1], blues[:, 2], color='blue')
        y_f = []
        for point in list(np.linspace(-1, 1)):
            y_f.append(self.applyLine(self.V, point))
        plt.plot(np.linspace(-1, 1), y_f, color='black', label='f(x)')
        if hasattr(self, 'w'):
            y_g = []
            for point in list(np.linspace(-1, 1)):
                y_g.append(self.applyLine(self.w, point))
            plt.plot(np.linspace(-1, 1), y_g, color='green', label='g(x)')
            plt.legend()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()
