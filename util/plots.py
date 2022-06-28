import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

def temp_vs_emissivity():
    bt = 315  # brightness temperature (Kelvin)
    lambda_ = 11e-6  # wavelength (m)
    rho = 1.438e-2  # scaling constant, (m Kelvin)

    ems = np.linspace(0.9, 1.0, 100)
    surface_temp = np.array([bt / (1 + ((lambda_ * bt / rho) * np.log(em))) for em in ems])

    plt.plot(ems, surface_temp, label=f'Brightness Temperature at {bt} K')
    plt.title('Surface Temperature vs. Emissivity')
    plt.xlabel('Emissivity')
    plt.ylabel('Surface Temperature')
    plt.xticks(np.arange(min(ems), max(ems)+0.01, 0.01))
    plt.yticks(np.arange(min(surface_temp), max(surface_temp), 1))
    plt.grid()
    plt.legend()
    plt.show()

def bt_vs_emissivity():
    ts = 300  # brightness temperature (Kelvin)
    lambda_ = 11e-6  # wavelength (m)
    rho = 1.438e-2  # scaling constant, (m Kelvin)

    ems = np.linspace(0.9, 1.0, 100)
    bt = ts + ts * lambda_ * b

def calc_surface_temp():
    bt = 300  # brightness temperature (Kelvin)
    rho = 1.438e-2  # scaling constant, (m Kelvin)
    em = 0.966
    surface_temp = bt / (1 + ((lambda_ * bt / rho) * np.log(em)))
    print(surface_temp)


def main():
    # mpl.use('Qt5Agg')
    temp_vs_emissivity()
    # calc_surface_temp()

if __name__ == '__main__':
    main()