import numpy as np


def main():
    a = np.array([1, 2, 3, 0, 0])
    print(np.average(a[a != 0]))


if __name__ == '__main__':
    main()