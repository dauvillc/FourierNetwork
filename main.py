import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class FourierSeriesNet(nn.Module):
    """
    A Neural Network for regression that approximates regular functions
    with linear combinations of trigonometric functions.
    """
    def __init__(self, base_frequency: float, harmonics: int):
        """
        :param base_frequency: Frequency of the first harmonic.
        :param harmonics: Number of Fourier harmonics used for the approximation
        """
        super(FourierSeriesNet, self).__init__()
        self.harmonics = harmonics
        self.base_frequency = base_frequency

        self.fc1 = nn.Linear(2 * harmonics, 1, bias=False)

    def forward(self, x_values: torch.Tensor):
        with torch.no_grad():
            # compute multiples of the base pulse: 2 * pi * f0 * k * x
            freqs = torch.ones((x_values.size()[0], self.harmonics), requires_grad=True) * x_values.view((-1, 1)) * 2 * np.pi * self.base_frequency
            for i in range(2, self.harmonics):
                freqs[:, i - 1] *= i

            # Apply sines and cosines
            cosVals = torch.cos(freqs)
            sinVals = torch.sin(freqs)
            ipt = torch.cat((cosVals, sinVals), 1)

        # Concatenate to obtain the trigonometric input
        # [cos(2 pi f x), cos(4 pi f x), cos(6 pi f x), ..., sin(2 pi f x), ...]
        # ipt = torch.cat((cosVals, sinVals), 1)

        # Apply the network's FC layers
        ipt = self.fc1(ipt)

        return ipt

@torch.no_grad()
def f(x: torch.Tensor):
    """
    Periodic function the network should approximate.
    :param x: Batch of data of shape (batch_size, 1)
    """
    # Some examples of 2-pi-periodic functions
    # return torch.cos(x.clone().detach())
    # return sum((torch.cos(i * x) + torch.sin(i * x) for i in range(3)))
    # return torch.cos(x) + 2.7 * torch.sin(3 * x) + 2 * torch.cos(2 * x) + torch.cos(4 * x) + torch.sin(9 * x)
    return torch.cos(x) * torch.sin(x) + torch.sin(7 * x) + torch.cos(6 * x + 2.3)


if __name__ == '__main__':
    # Create the net
    # Parameters:
    harmonics = 10
    base_freq = 1 / (2 * np.pi)
    net = FourierSeriesNet(base_freq, harmonics).double()

    # Optimizer and Loss Function
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    net.zero_grad()
    loss_func = nn.MSELoss()

    # Training parameters
    epochs = 150
    batch_size = 32

    # Memorize the error to plot it later on
    errors = torch.empty(epochs)

    # Training loop
    for i in range(epochs):
        optimizer.zero_grad()
        # Random input
        x = torch.rand((batch_size, 1)).double() * np.pi * 2
        output = net(x)
        target = f(x)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        errors[i] = loss.item()

    # Plotting the loss as well as the plot approximated by the network after the training
    with torch.no_grad():
        x = torch.from_numpy(np.linspace(0, np.pi * 2, 100)).double()

        plt.figure()

        axes = plt.subplot(221)
        # Network approximated plot
        plt.plot(x, net(x))
        # Real curve
        plt.plot(x, f(x), "g")
        plt.title("Approximation on the trained interval")
        plt.legend(["Network approximation", "Real curve"])

        plt.subplot(222)
        plt.title("MSE Loss")
        plt.xlabel("Epochs")
        plt.plot([i for i in range(epochs)], errors, "r-")

        # Plotting the network's approximation on an interval on which it hasn't trained
        plt.subplot(223)
        x = torch.from_numpy(np.linspace(np.pi, np.pi * 1.5, 100)).double()
        plt.plot(x, net(x))
        # Real curve
        plt.plot(x, f(x), "g")
        plt.title("Approximation on a non-trained interval")
        plt.legend(["Network approximation", "Real curve"])


        plt.show()

