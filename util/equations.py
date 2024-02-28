__author__ = 'yuhao liu'
"""
Physics related equations
"""


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