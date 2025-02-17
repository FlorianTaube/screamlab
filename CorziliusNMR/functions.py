import numpy as np
import re


def calc_exponential(time_vals, param):
    return list(param[0] * (1 - np.exp(-np.asarray(time_vals) / param[1])))


def calc_biexponential(time_vals, param):
    return list(
        param[0] * (1 - np.exp(-np.asarray(time_vals) / param[2]))
        + param[1] * (1 - np.exp(-np.asarray(time_vals) / param[3]))
    )


def calc_exponential_with_offset(time_vals, param):
    return list(
        param[0]
        * (1 - np.exp(-(np.asarray(time_vals) - param[2]) / param[1]))
    )


def calc_biexponential_with_offset(time_vals, param):
    return list(
        param[0]
        * (1 - np.exp(-(np.asarray(time_vals) - param[4]) / param[2]))
        + param[1]
        * (1 - np.exp(-(np.asarray(time_vals) - param[4]) / param[3]))
    )


def generate_spectra_param_dict(params):
    param_value_list = []
    for param in params:
        parts = re.split(r"_(cen|amp|sigma|gamma)_", param)
        if prefix != parts[0]:
            if param_value_list:
                param_dict[dict_index].append(param_value_list)
            prefix = parts[0]
            param_value_list = []
        if lastfix != parts[2]:
            if param_value_list:
                param_dict[dict_index].append(param_value_list)
                param_value_list = []
            lastfix = parts[2]
            dict_index += 1
        if dict_index not in param_dict:
            param_dict[dict_index] = []
        param_value_list.append(float(params[param].value))
        if parts[1] == "gamma":
            param_value_list.append("gam")
    param_dict[dict_index].append(param_value_list)
    return param_dict
