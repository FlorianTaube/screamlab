"""
fitting module of the CorziliusNMR package.
"""
from CorziliusNMR import utils
import sys

def scream_buildup_time_evaluation(output_file, peak_list,plus_minus_list=["+"], procpars=["103"],autopeakpick=False, hilfe=["max_value","voigt","global_voigt"]):
    if len(procpars) != 1: sys.exit("ERROR: Too many procpars.")
    csv_file_name = utils.generate_csv(output_file,procpars)
    delay_times = utils.get_delay_times_from_csv(csv_file_name)
    x_data, y_data = utils.read_xy_data_from_csv(csv_file_name)
    sys.exit() if not utils.check_if_plus_minus_list_contains_just_plus_and_minus(
        plus_minus_list) else None

    if autopeakpick:
        if utils.check_if_autopeakpick_is_possible(plus_minus_list):
            peak_list = utils.auto_peak_pick(peak_list, plus_minus_list, x_data[
                -1], y_data[-1])
        else:
            print(
                "ERROR: Auto peak picking is not possible if some peaks are positive and some are negative.")

    peak_infos = utils.generate_peak_dict(peak_list, plus_minus_list)
    peak_infos = utils.add_peak_label(peak_infos, procpars)
    for type in hilfe:
        if type == "max_value":
            intensitys = utils.get_intensitys_from_maximum(x_data, y_data,
                                                           peak_infos)
        elif type == "voigt":
            intensitys = utils.get_intensitys_from_voigt_fittings(x_data,
                                                                  y_data,
                                                                  peak_infos)
        elif type == "global_voigt":
            fitting_together = [[1], [2, 3, 4, 5]]
            intensitys = utils.get_intensitys_from_global_voigt_fittings(
                x_data, y_data, peak_infos, fitting_together, output_file)

        else:
            sys.exit("Unknown fitting type!")
        utils.calc_buildup(intensitys, delay_times, output_file, type,
                           fitting_type_list=["Biexponential"])