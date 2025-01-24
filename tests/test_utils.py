import CorziliusNMR.settings
import lmfit
import matplotlib.pyplot as plt
from CorziliusNMR import dataset, settings, utils
import unittest
import numpy as np


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.ds = dataset.Dataset()
        self.peak = dataset.Peak()
        self.fitter = utils.GlobalSpectrumFitter(self.ds)

    def add_n_spectra(self, number_of_spectra):
        for spec in range(0, number_of_spectra):
            self.ds.spectra.append(CorziliusNMR.dataset.Spectra())
            self.ds.add_peak(150)
        self.add_x_axis()
        self.add_y_axix()

    def add_x_axis(self):
        for spec in self.ds.spectra:
            spec.x_axis = np.linspace(100, 350, 1000)

    def add_y_axix(self):
        for spec_nr, spec in enumerate(self.ds.spectra):
            spec.y_axis = utils.voigt_profile(
                spec.x_axis, 250, 2, 2, (spec_nr + 1) * 200
            )

    def test_global_spectrum_fitter_init_dataset(self):
        self.assertEqual(
            type(self.fitter.dataset), CorziliusNMR.dataset.Dataset
        )

    def test_utils_fitter_get_axis_test_x(self):
        self.add_n_spectra(2)
        x_axis, y_axis = self.fitter._get_axis()
        self.assertTrue(np.array_equal(x_axis, np.linspace(100, 350, 1000)))

    def test_utils_fitter_get_axis_test_y(self):
        self.add_n_spectra(2)
        sim_axis = utils.voigt_profile(
            self.ds.spectra[0].x_axis, 250, 2, 2, (0 + 1) * 200
        )
        x_axis, y_axis = self.fitter._get_axis()
        self.assertTrue(np.array_equal(y_axis, sim_axis))

    def test_setup_params_return_val_is_lmfit_Parameter(self):
        self.assertEqual(
            type(self.fitter._setup_params("prefit")), lmfit.Parameters
        )

    def test_setup_params_add_peak_voigt_prefit(self):
        self.add_n_spectra(1)
        self.fitter.dataset.peak_list[0].buildup_type = "voigt"
        params = self.fitter._setup_params(7)
        print(params)
        self.assertEqual(
            params.keys,
            "Parameters([('Peak_at_150_ppm_amp_7', <Parameter 'Peak_at_150_ppm_amp_7', value=200, bounds=[0:inf]>), ('Peak_at_150_ppm_cen_7', <Parameter 'Peak_at_150_ppm_cen_7', value=150.0, bounds=[149.0:151.0]>), ('Peak_at_150_ppm_sig_7', <Parameter 'Peak_at_150_ppm_sig_7', value=10.0, bounds=[0:20]>), ('Peak_at_150_ppm_gam_7', <Parameter 'Peak_at_150_ppm_gam_7', value=10.0, bounds=[0:20]>)])",
        )
