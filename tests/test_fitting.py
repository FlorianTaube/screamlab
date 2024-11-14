import sys
from CorziliusNMR import io, fitting, utils,fitting_parameter_dicts
import numpy as np
import unittest
from lmfit.models import ExpressionModel
import matplotlib.pyplot as plt

class TestAnalysis(unittest.TestCase):
    def test_generate_csv(self):
        self.assertEqual(utils.generate_csv("HN-P-OH-100K","103"),"HN-P-OH-100K_103.csv" )

    def test_get_delay_times_csv(self):
        csv = r"C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\HN-P-100K_103.csv"
        self.assertEqual(utils.get_delay_times_from_csv(csv),[1,4,8,16,32,64,128,256,512,1024])

    def test_read_xy_data_from_csv(self):
        csv_file_name = r"C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\HN-P-100K_103.csv"
        data = np.loadtxt(csv_file_name, delimiter=",")
        y_data = [data[:, i] for i in range(0, np.size(data, 1)) if i % 2 != 0]
        self.assertTrue(np.array_equal(utils.read_xy_data_from_csv(csv_file_name)[1],y_data))

    def test_generate_peak_dict_1(self):
        self.assertEqual(utils.generate_peak_dict([172],["+"]),{172: ['+']})

    def test_generate_peak_dict_2(self):
        self.assertEqual(utils.generate_peak_dict([172,60,50,40,30],["+"]*5),{172: ['+'], 60: ['+'], 50: ['+'], 40: ['+'], 30: ['+']})

    def test_generate_peak_dict_3(self):
        self.assertEqual(utils.generate_peak_dict([172,60,50,40,30],["+"]),{172: ['+'], 60: ['+'], 50: ['+'], 40: ['+'], 30: ['+']})

    def test_generate_peak_dict_4(self):
        self.assertEqual(utils.generate_peak_dict([172,60,50,40,30],["+"]*12),{172: ['+'], 60: ['+'], 50: ['+'], 40: ['+'], 30: ['+']})

    def test_check_if_autopeakpick_is_possible_1(self):
        self.assertTrue(utils.check_if_autopeakpick_is_possible(["+"]*6))

    def test_check_if_autopeakpick_is_possible_2(self):
        self.assertTrue(utils.check_if_autopeakpick_is_possible(["-"]))

    def test_check_if_autopeakpick_is_possible_3(self):
        self.assertFalse(utils.check_if_autopeakpick_is_possible(["-","+"]))

    def test_check_if_plus_minus_list_contains_just_plus_and_minus_1(self):
        self.assertTrue(utils.check_if_plus_minus_list_contains_just_plus_and_minus(["+","-"]*6))

    def test_check_if_plus_minus_list_contains_just_plus_and_minus_2(self):
        self.assertFalse(utils.check_if_plus_minus_list_contains_just_plus_and_minus(["+","a"]*6))

    def test_auto_peak_pick_sim_data(self):
        x_sim = np.linspace(0, 200, 1000)
        gauss_params = [
            (1.0, 40, 5),
            (0.8, 80, 7),
            (1.2, 100, 10),
            (0.5, 140, 3),
            (1.0, 160, 8)
        ]
        y_sim = np.zeros_like(x_sim)
        for amp, center, width in gauss_params:
            y_sim += utils.gaussian(x_sim, amp, center, width)*100
        self.assertListEqual(utils.auto_peak_pick([10,20,30,40,50],["+"]*5,x_sim,y_sim),[40, 83, 99, 140, 159])

    def test_auto_peak_pick_sim_data_minus(self):
        x_sim = np.linspace(0, 200, 1000)
        gauss_params = [
            (-1.0, 40, 5),
            (-0.8, 80, 7),
            (-1.2, 100, 10),
            (-0.5, 140, 3),
            (-1.0, 160, 8)
        ]
        y_sim = np.zeros_like(x_sim)
        for amp, center, width in gauss_params:
            y_sim += utils.gaussian(x_sim, amp, center, width)*100
        self.assertListEqual(utils.auto_peak_pick([10,20,30,40,50],["-"]*5,x_sim,y_sim),[40, 83, 99, 140, 159])

    def test_auto_peak_pick_real_data(self):
        csv_file_name = r"C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\HN-P-100K_103.csv"
        data = np.loadtxt(csv_file_name, delimiter=",")
        x_data = [data[:, i] for i in range(0, np.size(data, 1)) if i % 2 == 0]
        y_data = [data[:, i] for i in range(0, np.size(data, 1)) if i % 2 != 0]
        peak_list = [172,60,50,40,30]
        plus_minus_list = ["-"]*5
        self.assertListEqual(utils.auto_peak_pick(peak_list, plus_minus_list, x_data[-1], y_data[-1]),
                             [172, 58, 44, 27, 22])

    def test_add_peak_label(self):
        peak_info = {172: ['+'], 60: ['+'], 50: ['+'], 40: ['+'], 30: ['+']}
        peak_info_result ={172: ['+', 'Peak_at_172_ppm_procno_103'], 60: ['+', 'Peak_at_60_ppm_procno_103'], 50: ['+', 'Peak_at_50_ppm_procno_103'], 40: ['+', 'Peak_at_40_ppm_procno_103'], 30: ['+', 'Peak_at_30_ppm_procno_103']}
        procpars = ["103"]
        peak_info = utils.add_peak_label(peak_info,procpars)
        self.assertEqual(peak_info,peak_info_result)

    def test_get_intensitys_from_maximum_real_data(self):
        csv_file_name = r"C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\HN-P-100K_103.csv"
        data = np.loadtxt(csv_file_name, delimiter=",")
        procpars =["103"]
        x_data = [data[:, i] for i in range(0, np.size(data, 1)) if i % 2 == 0]
        y_data = [data[:, i] for i in range(0, np.size(data, 1)) if i % 2 != 0]
        peak_list = [172,60,50,40,30]
        plus_minus_list = ["-"]*5
        peak_list = utils.auto_peak_pick(peak_list, plus_minus_list, x_data[-1], y_data[-1])
        peak_infos = utils.generate_peak_dict(peak_list, plus_minus_list)
        peak_infos = utils.add_peak_label(peak_infos, procpars)
        intensitys = utils.get_intensitys_from_maximum(x_data,y_data,peak_infos)
        intensitys_result = {'Peak_at_172_ppm_procno_103': [-99.61065, -667.68649, -1310.74418, -2090.19827, -2853.08531, -3713.76944, -4632.85067, -5654.55168, -6753.50623, -7804.47421], 'Peak_at_58_ppm_procno_103': [-121.53285, -867.04752, -1602.14841, -2392.12333, -3169.77973, -4008.02329, -4998.64478, -5953.05843, -7036.63094, -8080.62963], 'Peak_at_44_ppm_procno_103': [-126.76204, -852.01656, -1549.56431, -2355.21271, -3140.16258, -3968.12164, -4828.97639, -5817.89337, -6771.20514, -7852.20798], 'Peak_at_27_ppm_procno_103': [-117.56976, -863.88925, -1643.41392, -2460.42163, -3254.23405, -4129.1241, -5053.15608, -6027.93344, -7217.66349, -8138.45435], 'Peak_at_22_ppm_procno_103': [-144.2346, -880.16016, -1611.77438, -2387.03652, -3129.10513, -3926.98576, -4740.28355, -5633.73386, -6508.98015, -7562.16348]}
        self.assertEqual(intensitys,intensitys_result)

    def test_calc_buildup(self):
        intensitys_result = {
            'Peak_at_172_ppm_procno_103': [-99.61065, -667.68649, -1310.74418, -2090.19827, -2853.08531, -3713.76944,
                                           -4632.85067, -5654.55168, -6753.50623, -7804.47421],
            'Peak_at_58_ppm_procno_103': [-121.53285, -867.04752, -1602.14841, -2392.12333, -3169.77973, -4008.02329,
                                          -4998.64478, -5953.05843, -7036.63094, -8080.62963],
            'Peak_at_44_ppm_procno_103': [-126.76204, -852.01656, -1549.56431, -2355.21271, -3140.16258, -3968.12164,
                                          -4828.97639, -5817.89337, -6771.20514, -7852.20798],
            'Peak_at_27_ppm_procno_103': [-117.56976, -863.88925, -1643.41392, -2460.42163, -3254.23405, -4129.1241,
                                          -5053.15608, -6027.93344, -7217.66349, -8138.45435],
            'Peak_at_22_ppm_procno_103': [-144.2346, -880.16016, -1611.77438, -2387.03652, -3129.10513, -3926.98576,
                                          -4740.28355, -5633.73386, -6508.98015, -7562.16348]}
        delay_times = [1,4,8,16,32,64,128,256,512,1024]
        fitting_type_list = ["Biexponential","Biexponential_with_offset"]
        output_file = r"C:\Users\Florian Taube\Desktop\Prolin_auswertung_Test\HN-P-100K"
        utils.calc_buildup(intensitys_result,delay_times,output_file,fitting_type_list=fitting_type_list)

    def test_get_model_expression(self):
        self.assertEqual(fitting_parameter_dicts.get_model_expression("Exponential"),"A1*(1-exp(-x/x1))")

    def test_get_model_expression_1(self):
        self.assertEqual(fitting_parameter_dicts.get_model_expression("Biexponential"),"A1*(1-exp(-x/x1))+A2*(1-exp(-x/x2))")

    def test_get_model_expression_1(self):
        self.assertEqual(fitting_parameter_dicts.get_model_expression("Biexponential_with_offset"),"A1*(1-exp(-(x-x0)/x1))+A2*(1-exp(-(x-x0)/x2))")

    def test_get_model_expression_1(self):
        self.assertEqual(fitting_parameter_dicts.get_model_expression("Exponential_with_offset"),"A1*(1-exp(-(x-x0)/x1))")


    def test_get_param_dict(self):
        self.assertEqual(fitting_parameter_dicts.get_param_dict("Exponential"),
                         {'A1': dict(value=10),
                          'x1': dict(value=5, min=0)})

    def test_get_param_dict(self):
        self.assertEqual(fitting_parameter_dicts.get_param_dict("Exponential_with_offset"),
                         {'A1': dict(value=10),
                          'x1': dict(value=5, min=0),
                          'x0': dict(value=0, min=-10, max=10)})
    def test_get_param_dict_1(self):
        self.assertEqual(fitting_parameter_dicts.get_param_dict("Biexponential"),
                         {'A1': dict(value=10),
                          'A2': dict(value=10),
                          'x1': dict(value=5, min=0),
                          'x2': dict(value=5, min=0)})

    def test_get_param_dict_1(self):
        self.assertEqual(fitting_parameter_dicts.get_param_dict("Biexponential_with_offset"),
                         {'A1': dict(value=10),
                          'A2': dict(value=10),
                          'x1': dict(value=5, min=0),
                          'x2': dict(value=5, min=0),
                          'x0': dict(value=0, min=-10, max=10)})

    def test_solomon_two_spins(self):
        x = np.linspace(1,100,1000)
        param_dict = {
            'P_h': 0,
            'rho_h': 23,
            'sigma_HC': -16,
            'P_c': 0,
            'P_c0': 400,
            'P_h0': 500,
            'rho_c': 10}

        y = utils.solomon_two_spin(x,**param_dict)
        plt.plot(x,y)
        plt.show()
        plt.close()

    def test_solomon_fit(self):
        procpars=["103"]
        hilfe = ["max_value"]
        autopeakpick = False
        peak_list=[-15]
        plus_minus_list = ["+"]
        csv_file_name = r"C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\HN-P-OH-100K_103.csv"
        output_file = r"C:\Users\Florian Taube\Desktop\Prolin_auswertung_Test\HN-P-OH-100K_103"
        delay_times = utils.get_delay_times_from_csv(csv_file_name)
        x_data, y_data = utils.read_xy_data_from_csv(csv_file_name)
        sys.exit() if not utils.check_if_plus_minus_list_contains_just_plus_and_minus(plus_minus_list) else None

        if autopeakpick:
            if utils.check_if_autopeakpick_is_possible(plus_minus_list):
                peak_list = utils.auto_peak_pick(peak_list, plus_minus_list, x_data, y_data)
            else:
                print("ERROR: Auto peak picking is not possible if some peaks are positive and some are negative.")

        peak_infos = utils.generate_peak_dict(peak_list, plus_minus_list)
        peak_infos = utils.add_peak_label(peak_infos, procpars)
        for type in hilfe:
            if type == "max_value":
                intensitys = utils.get_intensitys_from_maximum(x_data, y_data, peak_infos)
            elif type == "voigt":
                intensitys = utils.get_intensitys_from_voigt_fittings(x_data,y_data,peak_infos)
                sys.exit()
            elif type == "global_voigt":
                print(type)
            utils.calc_buildup(intensitys, delay_times, output_file,fitting_type_list=["Solomon"])

    def test_get_intensitys_from_voigt_fittings_real_data_one_peak(self):
        procpars = ["103"]
        hilfe = ["voigt","max_value","global_voigt"]
        autopeakpick = False
        peak_list = [-15]
        plus_minus_list = ["+"]
        csv_file_name = r"C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Ala-AMUPOL_60_30_10_0.csv"
        output_file = r"C:\Users\Florian Taube\Desktop\Prolin_auswertung_Test\Ala-AMUPOL_60_30_10_0"
        delay_times = utils.get_delay_times_from_csv(csv_file_name)
        x_data, y_data = utils.read_xy_data_from_csv(csv_file_name)
        sys.exit() if not utils.check_if_plus_minus_list_contains_just_plus_and_minus(plus_minus_list) else None

        if autopeakpick:
            if utils.check_if_autopeakpick_is_possible(plus_minus_list):
                peak_list = utils.auto_peak_pick(peak_list, plus_minus_list, x_data, y_data)
            else:
                print("ERROR: Auto peak picking is not possible if some peaks are positive and some are negative.")

        peak_infos = utils.generate_peak_dict(peak_list, plus_minus_list)
        peak_infos = utils.add_peak_label(peak_infos, procpars)
        for type in hilfe:
            if type == "max_value":
                intensitys = utils.get_intensitys_from_maximum(x_data, y_data, peak_infos)
            elif type == "voigt":
                intensitys = utils.get_intensitys_from_voigt_fittings(x_data, y_data, peak_infos)
            elif type == "global_voigt":
                print(type)
            else:
                sys.exit("Unknown fitting type!")
            print("Hello")
            utils.calc_buildup(intensitys, delay_times, output_file,type,
                               fitting_type_list=["Solomon"])

    def test_get_intensitys_from_voigt_fittings_real_data_five_peaks(self):
            procpars = ["103"]
            hilfe = ["voigt","max_value"]
            autopeakpick = True
            peak_list = [172,60,50,40,30]
            plus_minus_list = ["-"]*5
            csv_file_name = r"C:\Users\Florian " \
                            r"Taube\Documents\Programmierung\CorziliusNMR" \
                            r"\tests\SCREAM_Test_Files\HN-P-100K_103.csv"
            output_file = r"C:\Users\Florian " \
                          r"Taube\Desktop\Prolin_auswertung_Test\HN-P-100K"
            delay_times = utils.get_delay_times_from_csv(csv_file_name)
            x_data, y_data = utils.read_xy_data_from_csv(csv_file_name)
            sys.exit() if not utils.check_if_plus_minus_list_contains_just_plus_and_minus(plus_minus_list) else None

            if autopeakpick:
                if utils.check_if_autopeakpick_is_possible(plus_minus_list):
                    peak_list = utils.auto_peak_pick(peak_list,
                                                     plus_minus_list, x_data[
                                                         -1], y_data[-1])
                else:
                    print("ERROR: Auto peak picking is not possible if some peaks are positive and some are negative.")

            peak_infos = utils.generate_peak_dict(peak_list, plus_minus_list)
            peak_infos = utils.add_peak_label(peak_infos, procpars)
            for type in hilfe:
                if type == "max_value":
                    intensitys = utils.get_intensitys_from_maximum(x_data, y_data, peak_infos)
                elif type == "voigt":
                    intensitys = utils.get_intensitys_from_voigt_fittings(x_data, y_data, peak_infos)
                elif type == "global_voigt":
                    print(type)
                else:
                    sys.exit("Unknown fitting type!")
                utils.calc_buildup(intensitys, delay_times, output_file,type,
                                   fitting_type_list=["Biexponential"])

    def test_get_intensitys_from_global_voigt_fittings_real_data_five_peaks(
            self):
        procpars = ["103"]
        hilfe = ["global_voigt","voigt","max_value"]
        autopeakpick = True
        peak_list = [172, 60, 50, 40, 30]
        plus_minus_list = ["-"] * 5
        csv_file_name = r"C:\Users\Florian " \
                        r"Taube\Documents\Programmierung\CorziliusNMR" \
                        r"\tests\SCREAM_Test_Files\HN-P-100K_103.csv"
        output_file = r"C:\Users\Florian " \
                      r"Taube\Desktop\Prolin_auswertung_Test\HN-P-100K"
        delay_times = utils.get_delay_times_from_csv(csv_file_name)
        x_data, y_data = utils.read_xy_data_from_csv(csv_file_name)
        sys.exit() if not utils.check_if_plus_minus_list_contains_just_plus_and_minus(plus_minus_list) else None

        if autopeakpick:
            if utils.check_if_autopeakpick_is_possible(plus_minus_list):
                peak_list = utils.auto_peak_pick(peak_list, plus_minus_list, x_data[
                                                         -1], y_data[-1])
            else:
                print("ERROR: Auto peak picking is not possible if some peaks are positive and some are negative.")

        peak_infos = utils.generate_peak_dict(peak_list, plus_minus_list)
        peak_infos = utils.add_peak_label(peak_infos, procpars)
        for type in hilfe:
            if type == "max_value":
                intensitys = utils.get_intensitys_from_maximum(x_data, y_data, peak_infos)
            elif type == "voigt":
                intensitys = utils.get_intensitys_from_voigt_fittings(x_data, y_data, peak_infos)
            elif type == "global_voigt":
                fitting_together = [[1],[2,3,4,5]]
                intensitys = utils.get_intensitys_from_global_voigt_fittings(
                    x_data, y_data, peak_infos,fitting_together,output_file)

            else:
                sys.exit("Unknown fitting type!")
            utils.calc_buildup(intensitys, delay_times, output_file,type,
                               fitting_type_list=["Biexponential"])

    def test_voigt_objective(self):
        params = []
        x_data = [np.array([0,1,2,3,4,5]),np.array([1,2,3,4,5,6])]
        y_data = [np.array([0, 1, 2, 3, 4, 5])*6, np.array([1, 2, 3, 4, 5,6])*2]
        result = utils.voigt_objective(params,x_data,y_data)
        self.assertEqual(sum(result), 96)


