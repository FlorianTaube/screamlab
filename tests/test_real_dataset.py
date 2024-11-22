import unittest

import CorziliusNMR.dataset


class TestDataset(unittest.TestCase):

    def test_first_dataset(self):
        dataset = CorziliusNMR.dataset.Dataset()
        dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
        dataset.expno_of_topspin_experiment =[22,33]
        dataset.procno_of_topspin_experiment = "103"
        dataset.output_file_name = r"C:\Users\Florian " \
                                   r"Taube\Desktop\Prolin_auswertung\Prolin_100K"
        dataset.peak_dict = {
            '173': dict(sign="-", fitting_group=2, fitting_model="voigt"),
            '58': dict(sign="-", fitting_group=4, fitting_model="voigt"),
            '44': dict(sign="-", fitting_group=4, fitting_model="voigt"),
            '28': dict(sign="-", fitting_group=4, fitting_model="voigt"),
            '22': dict(sign="-", fitting_group=4, fitting_model="voigt"),
        }
        dataset.start_buildup_fit_from_topspin_export()

    def test_first_dataset_1(self):
        dataset = CorziliusNMR.dataset.Dataset()
        dataset.path_to_topspin_experiment = r"F:\ssNMR\20241119_S31F_150K"
        dataset.expno_of_topspin_experiment =[24,32]
        dataset.procno_of_topspin_experiment = "103"
        dataset.output_file_name = r"C:\Users\Florian " \
                                   r"Taube\Desktop\Prolin_auswertung\Prolin_150K_natural_abundande"
        dataset.peak_dict = {
            '141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
            '27': dict(sign="+", fitting_group=1, fitting_model="voigt"),
            '13': dict(sign="+", fitting_group=3, fitting_model="voigt"),
            '-3': dict(sign="+", fitting_group=4, fitting_model="voigt"),
            '-8': dict(sign="+", fitting_group=4, fitting_model="voigt"),
        }
        dataset.start_buildup_fit_from_topspin_export()

    def test_first_dataset_2(self):
        dataset = CorziliusNMR.dataset.Dataset()
        dataset.path_to_topspin_experiment = r"F:\ssNMR\20241118_S31F_125K"
        dataset.expno_of_topspin_experiment =[24,32]
        dataset.procno_of_topspin_experiment = "103"
        dataset.output_file_name = r"C:\Users\Florian " \
                                   r"Taube\Desktop\Prolin_auswertung\Prolin_125K_natural_abundande"
        dataset.peak_dict = {
            '141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
            '27': dict(sign="+", fitting_group=4, fitting_model="voigt"),
            '13': dict(sign="+", fitting_group=4, fitting_model="voigt"),
            '-3': dict(sign="+", fitting_group=4, fitting_model="voigt"),
            '-8': dict(sign="+", fitting_group=4, fitting_model="voigt"),
        }
        dataset.start_buildup_fit_from_topspin_export()


