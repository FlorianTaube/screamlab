import unittest

import lmfit
from CorziliusNMR import settings, dataset


class TestDataset(unittest.TestCase):
    # TODO Write assertEQUALS
    def test_alanine_one_peak(self):
        props = settings.Properties()
        props.prefit = True
        props.spectrum_for_prefit = -2
        props.buildup_types = [
            "exponential",
            "exponential_with_offset",
            "biexponential",
        ]
        props.expno = [1, 8]
        props.procno = 103
        props.path_to_experiment = r"..\tests\SCREAM_Test_Files\Alanin"
        props.output_folder = r"..\tests\SCREAM_Test_Files\Alanin\result"
        ds = dataset.Dataset()
        ds.props = props
        ds.add_peak(-16)
        ds.start_buildup_fit_from_topspin()

    def test_prolin_one_peak(self):
        props = settings.Properties()
        props.prefit = True
        props.spectrum_for_prefit = -1
        props.buildup_types = [
            "biexponential",
        ]
        props.expno = [29, 32]
        props.output_folder = r"..\tests\SCREAM_Test_Files\Prolin\result"
        props.path_to_experiment = r"..\tests\SCREAM_Test_Files\Prolin"
        ds = dataset.Dataset()
        ds.props = props
        ds.add_peak(
            160, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}}
        )
        ds.add_peak(
            43, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}}
        )
        """        ds.add_peak(
            30, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}}
        )
        ds.add_peak(
            13, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}}
        )
        ds.add_peak(
            8, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}}
        )"""
        ds.start_buildup_fit_from_topspin()
