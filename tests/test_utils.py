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

    def add_n_spectra(self, number_of_spectra, type=["voigt"]):
        for spec in range(0, number_of_spectra):
            self.ds.spectra.append(CorziliusNMR.dataset.Spectra())
            self.ds.add_peak(150)
        self.add_x_axis()
        self.add_y_axix(type)

    def add_x_axis(self):
        for spec in self.ds.spectra:
            spec.x_axis = np.linspace(100, 350, 1000)

    def add_y_axix(self, type_list):
        for spec_nr, spec in enumerate(self.ds.spectra):
            y_axis = np.zeros(len(spec.x_axis))
            for type in type_list:
                if type == "voigt":
                    y_axis = y_axis + utils.voigt_profile(
                        spec.x_axis, 250, 2, 2, (spec_nr + 1) * 200
                    )
                if type == "gauss":
                    y_axis = y_axis + utils.gauss_profile(
                        spec.x_axis, 150, 3, (spec_nr + 1) * 200
                    )
                if type == "lorentz":
                    y_axis = y_axis + utils.lorentz_profile(
                        spec.x_axis, 200, 4, (spec_nr + 1) * 200
                    )
            spec.y_axis = y_axis

    def add_voigt_model_params(self):
        params = lmfit.Parameters()
        params.add("Peak_at_250_ppm_amp_0", value=200)
        params.add("Peak_at_250_ppm_cen_0", value=250)
        params.add("Peak_at_250_ppm_gam_0", value=2)
        params.add("Peak_at_250_ppm_sig_0", value=2)
        return params

    def add_gauss_model_params(self):
        params = lmfit.Parameters()
        params.add("Peak_at_150_ppm_amp_0", value=200)
        params.add("Peak_at_150_ppm_cen_0", value=150)
        params.add("Peak_at_150_ppm_sig_0", value=3)
        return params

    def add_lorentz_model_params(self):
        params = lmfit.Parameters()
        params.add("Peak_at_200_ppm_amp_0", value=200)
        params.add("Peak_at_200_ppm_cen_0", value=200)
        params.add("Peak_at_200_ppm_gam_0", value=4)
        return params

    def test_global_spectrum_fitter_init_dataset(self):
        self.assertEqual(
            type(self.fitter.dataset), CorziliusNMR.dataset.Dataset
        )

    def test_utils_fitter_get_axis_test_x(self):
        self.add_n_spectra(2)
        x_axis, y_axis = self.fitter._get_axis(1)
        self.assertTrue(np.array_equal(x_axis, np.linspace(100, 350, 1000)))

    def test_utils_fitter_get_axis_test_y(self):
        self.add_n_spectra(2)
        sim_axis = utils.voigt_profile(
            self.ds.spectra[0].x_axis, 250, 2, 2, (0 + 1) * 200
        )
        x_axis, y_axis = self.fitter._get_axis(0)
        self.assertTrue(np.array_equal(y_axis, sim_axis))

    def test_setup_params_return_val_is_list_of_lmfit_Parameters(self):
        result = self.fitter._setup_params("prefit")
        self.assertIsInstance(result, list)
        self.assertTrue(
            all(isinstance(param, lmfit.Parameters) for param in result)
        )

    def test_setup_params_add_peak_voigt_prefit(self):
        self.add_n_spectra(1)
        self.fitter.dataset.peak_list[0].fitting_type = "voigt"
        params = self.fitter._setup_params(7)
        self.assertEqual(
            str(params[0].keys()),
            "dict_keys(['Peak_at_150_ppm_amp_7', 'Peak_at_150_ppm_cen_7', 'Peak_at_150_ppm_sig_7', 'Peak_at_150_ppm_gam_7'])",
        )

    def test_setup_params_add_peak_lorentz_prefit(self):
        self.add_n_spectra(1)
        self.fitter.dataset.peak_list[0].fitting_type = "lorentz"
        params = self.fitter._setup_params(7)
        self.assertEqual(
            str(params[0].keys()),
            "dict_keys(['Peak_at_150_ppm_amp_7', 'Peak_at_150_ppm_cen_7', 'Peak_at_150_ppm_gam_7'])",
        )

    def test_setup_params_add_peak_gauss_prefit(self):
        self.add_n_spectra(1)
        self.fitter.dataset.peak_list[0].fitting_type = "gauss"
        params = self.fitter._setup_params(7)
        self.assertEqual(
            str(params[0].keys()),
            "dict_keys(['Peak_at_150_ppm_amp_7', 'Peak_at_150_ppm_cen_7', 'Peak_at_150_ppm_sig_7'])",
        )

    def test_setup_paramms_add_peak_gauss_negativ_peak(self):
        self.add_n_spectra(1)
        self.fitter.dataset.peak_list[0].fitting_type = "gauss"
        self.fitter.dataset.peak_list[0].peak_sign = "-"
        params = self.fitter._setup_params(7)
        bounds = [
            params[0]["Peak_at_150_ppm_amp_7"].min,
            params[0]["Peak_at_150_ppm_amp_7"].max,
        ]
        self.assertListEqual(bounds, [-np.inf, 0])

    def test_setup_paramms_add_peak_gauss_positive_peak(self):
        self.add_n_spectra(1)
        self.fitter.dataset.peak_list[0].fitting_type = "gauss"
        self.fitter.dataset.peak_list[0].peak_sign = "+"
        params = self.fitter._setup_params(7)
        bounds = [
            params[0]["Peak_at_150_ppm_amp_7"].min,
            params[0]["Peak_at_150_ppm_amp_7"].max,
        ]
        self.assertListEqual(bounds, [0, np.inf])

    #########################
    def test_perform_prefit_one_peak_without_noise_voigt(self):
        pass

    def test_prefit_objective_with_voigt(self):
        self.add_n_spectra(1, type=["voigt"])
        x_axis, y_axis = self.fitter._get_axis(0)
        params = self.add_voigt_model_params()
        residual = self.fitter._prefit_objective([params], x_axis, y_axis)
        self.assertTrue(np.array_equal(residual, np.zeros(len(y_axis))))

    def test_prefit_objective_with_gauss(self):
        self.add_n_spectra(1, type=["gauss"])
        x_axis, y_axis = self.fitter._get_axis(0)
        params = self.add_gauss_model_params()
        residual = self.fitter._prefit_objective([params], x_axis, y_axis)
        self.assertTrue(np.array_equal(residual, np.zeros(len(y_axis))))

    def test_prefit_objective_with_lorentz(self):
        self.add_n_spectra(1, type=["lorentz"])
        x_axis, y_axis = self.fitter._get_axis(0)
        params = self.add_lorentz_model_params()
        residual = self.fitter._prefit_objective([params], x_axis, y_axis)
        self.assertTrue(np.array_equal(residual, np.zeros(len(y_axis))))

    def test_prefit_objective_with_lorentz_gauss_and_voigt(self):
        self.add_n_spectra(1, type=["voigt", "gauss", "lorentz"])
        x_axis, y_axis = self.fitter._get_axis(0)
        params_lorentz = self.add_lorentz_model_params()
        params_gauss = self.add_gauss_model_params()
        params_voigt = self.add_voigt_model_params()
        residual = self.fitter._prefit_objective(
            [params_lorentz, params_gauss, params_voigt], x_axis, y_axis
        )
        self.assertTrue(
            np.allclose(residual, np.zeros(len(y_axis)), atol=1e-9)
        )
