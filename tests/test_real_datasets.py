import unittest

import lmfit
from CorziliusNMR import settings, dataset


class TestDataset(unittest.TestCase):

    def test_alanine_one_peak(self):
        props = settings.Properties()
        props.prefit = True
        props.spectrum_for_prefit = -3
        props.buildup_types = ["exponential"]
        props.expno = [1, 8]
        props.procno = 103
        props.path_to_experiment = r"C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Alanin"
        ds = dataset.Dataset()
        ds.props = props
        ds.add_peak(-16)
        ds.start_buildup_fit_from_topspin()

    def test_prolin_one_peak(self):
        props = settings.Properties()
        props.prefit = True
        props.spectrum_for_prefit = -2
        props.buildup_types = ["biexponential"]
        props.expno = [24, 32]

        props.procno = 103
        props.path_to_experiment = r"C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files\Prolin"
        ds = dataset.Dataset()
        ds.props = props
        ds.add_peak(
            160, line_broadening={"sigma": {"max": 3}, "gamma": {"max": 3}}
        )
        ds.add_peak(
            43, line_broadening={"sigma": {"max": 3}, "gamma": {"max": 3}}
        )
        ds.add_peak(
            30, line_broadening={"sigma": {"max": 3}, "gamma": {"max": 3}}
        )
        ds.add_peak(
            13, line_broadening={"sigma": {"max": 3}, "gamma": {"max": 3}}
        )
        ds.add_peak(
            8, line_broadening={"sigma": {"max": 3}, "gamma": {"max": 3}}
        )
        ds.start_buildup_fit_from_topspin()
