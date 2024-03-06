__author__ = 'yuhao liu'
"""
Physics related equations
"""


def calc__broadband_emis_cheng(emis_10, emis_11, emis_12, emis_13, emis_14):
    """
    Calculate the broadband emissivity from the ASTER emissivity bands.
    Cheng, J., Liang, S., Yao, Y., Zhang, X., 2013.
    Estimating the optimal broadband emissivity spectral range for calculating surface longwave net radiation.
    IEEE Geosci. Remote Sens. Lett. 10, 401–405.
    :return:
    """
    broadband_emis = 0.197 + 0.025 * emis_10 + 0.057 * emis_11 + 0.237 * emis_12 + 0.333 * emis_13 + 0.146 * emis_14
    return broadband_emis


def calc_broadband_emis_ogawa(emis_10, emis_11, emis_12, emis_13, emis_14):
    """
    Calculate the broadband emissivity from the ASTER emissivity bands.
    Following K. Ogawa, T. Schmugge, and S. Rokugawa, “Estimating broadband emissivity of arid regions and its
    seasonal variations using thermal infrared remote sensing,” (in English),
    IEEE Trans. Geosci. Remote Sens., vol. 46, no. 2, pp. 334–343, Feb. 2008.
    :return:
    """
    # Calculate the broadband emissivity
    broadband_emis = 0.128 + 0.014 * emis_10 + 0.145 * emis_11 + 0.241 * emis_12 + 0.467 * emis_13 + 0.004 * emis_14
    return broadband_emis


def calc_lst(emis, uw_ir, dw_ir):
    """
    Calculate the land surface temperature (LST) using the upwelling and downwelling thermal infrared radiation.
    :param emis:
    :param uw_ir:
    :param dw_ir:
    :return:
    """
    sigma = 5.670374419e-8  # Stefan–Boltzmann constant
    lst = ((1 / (emis * sigma)) * (uw_ir - (1-emis) * dw_ir))**0.25
    return lst


def cvt_celsius_to_kelvin(celsius):
    """
    Convert Celsius to Kelvin
    :param celsius:
    :return:
    """
    return celsius + 273.15