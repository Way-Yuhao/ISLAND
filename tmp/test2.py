import numpy as np



def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def gker2(canvas, center, sig):
    x_length, y_length = canvas
    x_center, y_center = center
    x_ax = np.linspace(-x_center, x_length - x_center - 1, x_length)
    y_ax = np.linspace(-y_center, y_length - y_center - 1, y_length)
    x_gauss = np.exp(-0.5 * np.square(x_ax) / np.square(sig))
    y_gauss = np.exp(-0.5 * np.square(y_ax) / np.square(sig))
    kernel = np.outer(x_gauss, y_gauss)
    return kernel / np.sum(kernel)


def main():
    # kernel = gkern(l=2592*2, sig=100)
    # plt.imshow(kernel)
    # plt.show()

    # kernel = gker2(canvas=(2536, 2592), center=(1024, 1024), sig=500)
    # plt.imshow(kernel)
    #plt.show()
    x = 0.0234
    print(f"{x:.3%}")

if __name__ == '__main__':
    main()