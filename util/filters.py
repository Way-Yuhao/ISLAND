import numpy as np


def gkern_center(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    normalized = kernel / np.sum(kernel)
    return normalized.astype(np.float32)


def gkern(canvas, center, sig):
    x_length, y_length = canvas
    x_center, y_center = center
    x_ax = np.linspace(-x_center, x_length - x_center - 1, x_length, dtype=np.float32)
    y_ax = np.linspace(-y_center, y_length - y_center - 1, y_length, dtype=np.float32)
    x_gauss = np.exp(-0.5 * np.square(x_ax) / np.square(sig), dtype=np.float32)
    y_gauss = np.exp(-0.5 * np.square(y_ax) / np.square(sig), dtype=np.float32)
    kernel = np.outer(x_gauss, y_gauss)
    return kernel  # consider adding / np.sum(kernel)


def l2_kern(canvas, center, sig):
    x_length, y_length = canvas
    x_center, y_center = center
    x_ax = np.abs(np.linspace(-x_center, x_length - x_center - 1, x_length, dtype=np.float32))
    y_ax = np.abs(np.linspace(-y_center, y_length - y_center - 1, y_length, dtype=np.float32))
    xx, yy = np.meshgrid(x_ax, y_ax)
    kernel = np.sqrt(x_length ** 2. + y_length ** 2.) - np.sqrt(xx ** 2. + yy ** 2.)
    return kernel