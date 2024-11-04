import sys
from CorziliusNMR import io, fitting, SCREAM_mod_analyzer
import unittest
from lmfit.models import ExpressionModel

class TestAnalysis(unittest.TestCase):
    def test_generate_csv(self):
        self.assertEqual(fitting.generate_csv("Ala-AMUPOL_60_30_10_0.5"),"Ala-AMUPOL_60_30_10_0.5.csv" )

    def test_check_csv_name_for_backslash(self):
        self.assertEqual(fitting.check_csv_name_for_backslash("\\Yavin\\Hello"),"/Yavin/Hello")

    def test_get_delay_times_from_csv(self):
       csv = fitting.generate_csv(r'C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Ala-AMUPOL_60_30_10_0.5')
       csv = fitting.check_csv_name_for_backslash(csv)
       delay_times = fitting.get_delay_times_from_csv(csv)
       self.assertEqual(delay_times, [0.25,0.5,1,2,4,8,16])

    def test_xy_data_list_have_same_size(self):
        x_data, y_data = fitting.read_xy_data_from_csv(r'C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Ala-AMUPOL_60_30_10_0.5.csv')
        self.assertEqual(len(x_data),len(y_data))

    def test_xy_data_read_in(self):
        x_data, y_data = fitting.read_xy_data_from_csv(
            r'C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Ala-AMUPOL_60_30_10_0.5.csv')
        self.assertEqual(len(x_data), 7)

    def test_autopeakpick_possible_minus(self):
        self.assertTrue(fitting.check_if_autopeakpick_is_possible(["-","-","-","-","-","-","-"]))
    def test_autopeakpick_possible_plus(self):
        self.assertTrue(fitting.check_if_autopeakpick_is_possible(["+"]))
    def test_autopeakpick_possible_plusminus(self):
        self.assertFalse(fitting.check_if_autopeakpick_is_possible(["-", "-", "+", "-", "-", "+", "-"]))

    def test_auto_peak_pick(self):
        csv_file_name = r'C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\HN-P-100K.csv'
        x_data, y_data = fitting.read_xy_data_from_csv(csv_file_name)
        peak_list =[172,60,50,40,30]
        plus_minus_list = ["-"]*5
        peak_center = fitting.auto_peak_pick(peak_list,plus_minus_list,x_data,y_data)
        self.assertEqual(peak_center, [172, 58, 44, 27, 22])

    def test_auto_peak_pick_2(self):
        csv_file_name = r'C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Ala-AMUPOL_60_30_10_0.5.csv'
        x_data, y_data = fitting.read_xy_data_from_csv(csv_file_name)
        peak_list =[172]
        plus_minus_list = ["+"]
        peak_center = fitting.auto_peak_pick(peak_list,plus_minus_list,x_data,y_data)
        self.assertEqual(peak_center, [-15])

    def test_generate_peak_label_name(self):
        peak_lables, integral = fitting.generate_peak_label([172,-60])
        self.assertEqual(peak_lables, ['Peak_at_172_ppm', 'Peak_at_m60_ppm'])

    def test_generate_peak_label_integral_empty_list(self):
        peak_lables, integral = fitting.generate_peak_label([172,-60])
        self.assertEqual(integral, {'Peak_at_172_ppm': [], 'Peak_at_m60_ppm': []})

    def test_calculate_max_for_each_peak(self):
        csv_file_name = r'C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Ala-AMUPOL_60_30_10_0.5.csv'
        x_data, y_data = fitting.read_xy_data_from_csv(csv_file_name)
        peak_list = [172]
        plus_minus_list = ["+"]
        peak_lables, integral = fitting.generate_peak_label(peak_list)
        integral = fitting.calculate_max_for_each_peak(x_data,y_data,peak_lables,peak_list,integral,plus_minus_list)
        self.assertEqual(integral, {'Peak_at_172_ppm': [76.74219, 182.65625, 339.09375, 673.85938, 1356.39062, 2583.125, 4221.3125]})

    def test_calculate_max_for_each_peak_5_peaks(self):
        csv_file_name = r'C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\HN-P-100K.csv'
        x_data, y_data = fitting.read_xy_data_from_csv(csv_file_name)
        peak_list = [172,60,50,40,30]
        plus_minus_list = ["+"]*5
        peak_lables, integral = fitting.generate_peak_label(peak_list)
        integral = fitting.calculate_max_for_each_peak(x_data,y_data,peak_lables,peak_list,integral,plus_minus_list)
        self.assertEqual(integral,{'Peak_at_172_ppm': [30.50456, 35.99164, 37.99502, 41.46157, 41.44597, 39.70028, 58.65983, 147.97642, 123.02011, 149.16688], 'Peak_at_60_ppm': [33.48669, 31.58807, 36.67945, 41.71767, 45.22095, 44.6465, 78.81824, 99.54074, 123.24465, 135.23687], 'Peak_at_50_ppm': [34.10136, 34.8214, 37.62537, 40.85758, 37.64237, 37.5726, 63.21957, 101.42123, 126.49496, 134.20734], 'Peak_at_40_ppm': [32.54253, 32.16578, 36.95948, 45.71699, 41.81245, 42.1367, 64.77216, 110.79234, 133.9982, 148.1168], 'Peak_at_30_ppm': [40.97022, 42.68452, 33.19326, 40.27385, 40.78708, 72.60488, 102.22617, 118.80209, 182.41934]})

    def test_main_max_value(self):
        output_file =  r'C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Ala-AMUPOL_60_30_10_0'
        peak_list = [-15]
        fitting.main(output_file,peak_list,evaluation_type="max_value")

    def test_main_voigt(self):
        output_file =  r'C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Ala-AMUPOL_60_30_10_0'
        peak_list = [-15]
        fitting.main(output_file,peak_list,evaluation_type="voigt")

    def test_main_max_value_five_peaks(self):
        output_file =  r'C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Pro_Test'
        peak_list = [172,60,50,40,30]
        plus_minus = ["-"]*5
        fitting.main(output_file,peak_list,plus_minus_list=plus_minus,evaluation_type="max_value",fitting_type = ["Solomon_biexp"])

    def test_main_voigt_five_peaks(self):
        output_file =  r'C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Pro_Test'
        peak_list = [172,60,50,40,30]
        plus_minus = ["-"]*5
        fitting.main(output_file,peak_list,plus_minus_list=plus_minus,evaluation_type="voigt",fitting_type = ["Solomon_biexp"])

    def test_main_global_voigt_five_peaks(self):
        output_file =  r'C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Pro_Test'
        peak_list = [172,60,50,40,30]
        plus_minus = ["-"]*5
        fitting.main(output_file,peak_list,plus_minus_list=plus_minus,evaluation_type="global_voigt",fitting_type = ["Solomon_biexp"])

    def test_lhs_buildup_fit(self):
        integral = [0,2,4,6,8,10,12,14,16]
        delay_times = [0,2,4,6,8,10,12,14,16]
        fitting_type = "Solomon_biexp"
        model_expr = fitting.get_model_exp(fitting_type)
        buildup_fit_model = ExpressionModel(model_expr)
        param_dict = fitting.get_param_dict(fitting_type)
        fitting.lhs_buildup_fit(integral,buildup_fit_model, param_dict,delay_times)