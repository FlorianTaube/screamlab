import unittest

import lmfit
from CorziliusNMR import settings, dataset


class TestDataset(unittest.TestCase):

    def test_alanine_one_peak(self):
        props = settings.Properties()
        props.prefit = True
        props.spectrum_for_prefit = -2
        props.expno = [1, 7]
        props.procno = 103
        props.path_to_experiment = r"C:\Users\Florian Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files"
        ds = dataset.Dataset()
        ds.props = props
        ds.add_peak(-16)
        ds.start_buildup_fit_from_topspin()
