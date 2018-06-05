import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit
from matplotlib import cm

class activation_loss(object):
    def __init__(self):
        # set ax
        weights = np.arange(-5, 5, 0.5)
        bias = np.arange(-5, 5, 0.5)
        self.weights, self.bias = np.meshgrid(weights, bias)
        self.shape = weights.shape

        # set the input and output
        self.x = np.ones(self.shape, np.float64)
        self.y = 0.5 * np.ones(self.shape, np.float64)

    def plot_3d(self, zaxis, z):
        # plot the activations against the weight and bias
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X=self.weights, Y=self.bias, Z=z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title(zaxis)
        ax.set_xlabel('Weight')
        ax.set_ylabel('Bias')
        ax.set_zlabel(zaxis)
        plt.show()

    def q1(self):
        # compute sigmoid activations
        self.y_pred = expit(self.weights * self.x + self.bias)
        self.plot_3d('Sigmoid Activation', self.y_pred)

    def q2(self):
        l2_loss = (self.y_pred - self.y) ** 2
        self.plot_3d('L2-Loss', l2_loss)

    def q3(self):
        # compute gradient of L2-loss w.r.t. weight
        l2_gradient = 2 * (self.y_pred - self.y) * self.y_pred * (1-self.y_pred)*self.x
        self.plot_3d('L2-Loss Gradient', l2_gradient)

    def q4(self):
        # compute the cross-entropy loss
        ce_loss = -(self.y * np.log(self.y_pred) + (1. - self.y) * np.log(1. - self.y_pred))
        self.plot_3d('Cross Entropy Loss', ce_loss)

    def q5(self):
        ce_gradient = -self.x*(self.y-self.y_pred)
        self.plot_3d('Cross Entropy Loss gradient', ce_gradient)

def main():
    # run through Part 1
    p1 = activation_loss()
    p1.q1()
    p1.q2()
    p1.q3()
    p1.q4()
    p1.q5()

if __name__ =='__main__':
    main()