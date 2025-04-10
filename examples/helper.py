import numpy as np
import matplotlib.pyplot as plt


def plotvector(grid, values, title=""):
    x, y = grid.T.detach().numpy()
    Fx, Fy = values.T.detach().numpy()

    # Compute the amplitude (magnitude) of the vectors
    amplitude = np.sqrt(Fx**2 + Fy**2)

    # Reshape x, y, and amplitude for pcolormesh
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    X, Y = np.meshgrid(x_unique, y_unique)
    amplitude_grid = amplitude.reshape(len(y_unique), len(x_unique))

    # Normalize the vectors to make all arrows the same length
    Fx_normalized = Fx / amplitude
    Fy_normalized = Fy / amplitude

    # Plot the background amplitude
    plt.pcolormesh(X, Y, amplitude_grid, cmap="plasma", shading="auto")
    plt.colorbar(label="|F|")

    # Overlay the arrows
    plt.quiver(x, y, Fx_normalized, Fy_normalized, color="white", scale=40)

    plt.title(title)
    plt.xlim((min(x), max(x)))
    plt.ylim((min(y), max(y)))
    plt.show()


def plotscalar(grid, values, title=""):
    x, y = grid.T.detach().numpy()

    # Reshape x, y, and amplitude for pcolormesh
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    X, Y = np.meshgrid(x_unique, y_unique)
    amplitude_grid = values.detach().numpy().reshape(len(y_unique), len(x_unique))

    # Plot the background amplitude
    plt.pcolormesh(X, Y, amplitude_grid, cmap="plasma", shading="auto")
    plt.colorbar(label="|F|")

    plt.title(title)
    plt.xlim((min(x), max(x)))
    plt.ylim((min(y), max(y)))
    plt.show()
