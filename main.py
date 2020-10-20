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
    # return torch.cos(x.clone().detach())
    # return sum((torch.cos(i * x) + torch.sin(i * x) for i in range(3)))
    return torch.cos(x) + torch.sin(3 * x) + 2 * torch.cos(2 * x) + torch.cos(4 * x) + torch.sin(9 * x)


if __name__ == '__main__':
    net = FourierSeriesNet(1 / (2 * np.pi), 10).double()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    net.zero_grad()
    loss_func = nn.MSELoss()

    epochs = 150

    errors = torch.empty(epochs)
    for i in range(epochs):
        optimizer.zero_grad()
        x = torch.rand((32, 1)).double() * np.pi * 2
        output = net(x)
        target = f(x)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        errors[i] = loss.item()

    with torch.no_grad():
        x = torch.from_numpy(np.linspace(0, np.pi * 2, 100)).double()

        plt.figure()
        axes = plt.subplot(211)
        plt.plot(x, net(x))
        plt.plot(x, f(x), "g")
        plt.title("Réseau de neurones trigonométrique")
        plt.ylim(-3, 3)
        plt.subplot(212)
        plt.title("MSE Loss")
        plt.plot([i for i in range(epochs)], errors, "r-")
        plt.show()

