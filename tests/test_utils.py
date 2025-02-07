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
        self.fitter = utils.Fitter(self.ds)
        self.prefitter = utils.Prefitter(self.ds)
        self.globalfitter = utils.GlobalFitter(self.ds)
        self.singlefitter = utils.SingleFitter(self.ds)

    def assertListAlmostEqual(self, list1, list2, delta=1e-6):
        """Check if two lists are almost equal element-wise within a tolerance."""
        self.assertEqual(
            len(list1), len(list2), "Lists are of different lengths"
        )
        for i, (a, b) in enumerate(zip(list1, list2)):
            self.assertAlmostEqual(
                a, b, delta=delta, msg=f"Mismatch at index {i}: {a} != {b}"
            )

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

    def add_noise(self, y_axis):
        y_axis = np.array(y_axis)
        noise = np.random.normal(0, 1, size=y_axis.shape)
        return y_axis + noise

    def test_fitter_init_dataset(self):
        self.assertEqual(
            type(self.fitter.dataset), CorziliusNMR.dataset.Dataset
        )

    def test_prefitter_init_dataset(self):
        self.assertEqual(
            type(self.prefitter.dataset), CorziliusNMR.dataset.Dataset
        )

    def test_globalfitter_init_dataset(self):
        self.assertEqual(
            type(self.globalfitter.dataset), CorziliusNMR.dataset.Dataset
        )

    def test_singlefitter_init_dataset(self):
        self.assertEqual(
            type(self.singlefitter.dataset), CorziliusNMR.dataset.Dataset
        )

    def test_generate_x_axis_list_prefitter(self):
        self.add_n_spectra(1)
        x_axis, y_axis = self.prefitter._generate_axis_list()
        self.assertTrue(
            np.array_equal(x_axis[0], np.linspace(100, 350, 1000))
        )

    def test_generate_axis_list_same_lenght_prefitter(self):
        self.add_n_spectra(1)
        x_axis, y_axis = self.prefitter._generate_axis_list()
        self.assertEqual(len(x_axis[0]), len(y_axis[0]))

    def test_generate_x_axis_has_one_element_prefitter(self):
        self.add_n_spectra(1)
        x_axis, y_axis = self.prefitter._generate_axis_list()
        self.assertEqual(len(x_axis), 1)

    def test_generate_y_axis_has_one_element_prefitter(self):
        self.add_n_spectra(1)
        x_axis, y_axis = self.prefitter._generate_axis_list()
        self.assertEqual(len(y_axis), 1)

    def test_generate_y_axis_correct_spectrum_prefitter(self):
        self.add_n_spectra(1)
        x_axis, y_axis = self.prefitter._generate_axis_list()
        self.assertEqual(
            max(y_axis[0]),
            max(
                utils.voigt_profile(self.ds.spectra[0].x_axis, 250, 2, 2, 200)
            ),
        )

    def test_generate_y_axis_correct_spectrum_prefitter_with_more_spectra(
        self,
    ):
        self.add_n_spectra(7)
        self.prefitter.dataset.props.spectrum_for_prefit = 6
        x_axis, y_axis = self.prefitter._generate_axis_list()
        self.assertEqual(
            max(y_axis[0]),
            max(
                utils.voigt_profile(
                    self.ds.spectra[0].x_axis, 250, 2, 2, 7 * 200
                )
            ),
        )

    def test_generate_x_axis_global_fitter(self):
        self.add_n_spectra(7)
        x_axis, y_axis = self.globalfitter._generate_axis_list()
        self.assertEqual(len(x_axis), 7)

    def test_generate_y_axis_global_fitter(self):
        self.add_n_spectra(7)
        x_axis, y_axis = self.globalfitter._generate_axis_list()
        self.assertEqual(len(y_axis), 7)

    def test_generate_y_axis_global_fitter(self):
        self.add_n_spectra(7)
        x_axis, y_axis = self.globalfitter._generate_axis_list()
        maxima = []
        sim_maxima = []
        for nr, axis in enumerate(y_axis):
            maxima.append(max(axis))
            sim_maxima.append(
                max(
                    utils.voigt_profile(
                        self.ds.spectra[0].x_axis, 250, 2, 2, (nr + 1) * 200
                    )
                )
            )
        self.assertListEqual(maxima, sim_maxima)

    def test_generate_x_axis_single_fitter(self):
        self.add_n_spectra(7)
        x_axis, y_axis = self.singlefitter._generate_axis_list()
        self.assertEqual(len(x_axis), 7)

    def test_generate_y_axis_single_fitter(self):
        self.add_n_spectra(6)
        x_axis, y_axis = self.singlefitter._generate_axis_list()
        self.assertEqual(len(y_axis), 6)

    def test_generate_y_axis_single_fitter(self):
        self.add_n_spectra(6)
        x_axis, y_axis = self.globalfitter._generate_axis_list()
        maxima = []
        sim_maxima = []
        for nr, axis in enumerate(y_axis):
            maxima.append(max(axis))
            sim_maxima.append(
                max(
                    utils.voigt_profile(
                        self.ds.spectra[0].x_axis, 250, 2, 2, (nr + 1) * 200
                    )
                )
            )
        self.assertListEqual(maxima, sim_maxima)

    def test_get_amplitude_dict_positive_sign_prefitter(self):
        self.add_n_spectra(1)
        value = self.prefitter._get_amplitude_dict(self.ds.peak_list[0], 0)
        self.assertDictEqual(
            value,
            {
                "name": "Peak_at_150_ppm_amp_0",
                "value": 200,
                "min": 0,
                "max": np.inf,
            },
        )

    def test_get_amplitude_dict_negative_sign_prefitter(self):
        self.add_n_spectra(1)
        self.ds.peak_list[0].peak_sign = "-"
        value = self.prefitter._get_amplitude_dict(self.ds.peak_list[0], 0)
        self.assertDictEqual(
            value,
            {
                "name": "Peak_at_150_ppm_amp_0",
                "value": -200,
                "min": -np.inf,
                "max": 0,
            },
        )

    def test_get_center_dict_prefitter(self):
        self.add_n_spectra(1)
        value = self.prefitter._get_center_dict(self.ds.peak_list[0], 0)
        self.assertDictEqual(
            value,
            {
                "name": "Peak_at_150_ppm_cen_0",
                "value": 150,
                "min": 149,
                "max": 151,
            },
        )

    def test_get_lw_dict_sigma_prefitter(self):
        self.add_n_spectra(1)
        value = self.prefitter._get_lw_dict(self.ds.peak_list[0], 0, "sigma")
        self.assertDictEqual(
            value,
            {
                "name": "Peak_at_150_ppm_sigma_0",
                "value": 10,
                "min": 0,
                "max": 20,
            },
        )

    def test_get_lw_dict_gamma_prefitter(self):
        self.add_n_spectra(1)
        value = self.prefitter._get_lw_dict(self.ds.peak_list[0], 0, "gamma")
        self.assertDictEqual(
            value,
            {
                "name": "Peak_at_150_ppm_gamma_0",
                "value": 10,
                "min": 0,
                "max": 20,
            },
        )

    def test_get_lw_dict_gamma_non_default_values_prefitter(self):
        self.add_n_spectra(1)
        self.ds.peak_list[0].line_broadening["gamma"]["max"] = 5
        value = self.prefitter._get_lw_dict(self.ds.peak_list[0], 0, "gamma")
        self.assertDictEqual(
            value,
            {
                "name": "Peak_at_150_ppm_gamma_0",
                "value": 2.5,
                "min": 0,
                "max": 5,
            },
        )

    def test_generate_params_list_one_voigt_prefitter(self):
        self.add_n_spectra(1)
        params = self.prefitter._generate_params_list()
        self.assertListEqual(
            list(params.keys()),
            [
                "Peak_at_150_ppm_amp_0",
                "Peak_at_150_ppm_cen_0",
                "Peak_at_150_ppm_sigma_0",
                "Peak_at_150_ppm_gamma_0",
            ],
        )

    def test_generate_params_list_one_gauss_prefitter(self):
        self.add_n_spectra(1)
        self.ds.peak_list[0].fitting_type = "gauss"
        params = self.prefitter._generate_params_list()
        self.assertListEqual(
            list(params.keys()),
            [
                "Peak_at_150_ppm_amp_0",
                "Peak_at_150_ppm_cen_0",
                "Peak_at_150_ppm_sigma_0",
            ],
        )

    def test_generate_params_list_one_lorentz_prefitter(self):
        self.add_n_spectra(1)
        self.ds.peak_list[0].fitting_type = "lorentz"
        params = self.prefitter._generate_params_list()
        self.assertListEqual(
            list(params.keys()),
            [
                "Peak_at_150_ppm_amp_0",
                "Peak_at_150_ppm_cen_0",
                "Peak_at_150_ppm_gamma_0",
            ],
        )

    def test_generate_params_list_two_voigt_prefitter(self):
        self.add_n_spectra(1)
        self.ds.add_peak(120)
        params = self.prefitter._generate_params_list()
        self.assertListEqual(
            list(params.keys()),
            [
                "Peak_at_150_ppm_amp_0",
                "Peak_at_150_ppm_cen_0",
                "Peak_at_150_ppm_sigma_0",
                "Peak_at_150_ppm_gamma_0",
                "Peak_at_120_ppm_amp_0",
                "Peak_at_120_ppm_cen_0",
                "Peak_at_120_ppm_sigma_0",
                "Peak_at_120_ppm_gamma_0",
            ],
        )

    def test_generate_params_list_voigt_gauss_lorentz_prefitter(self):
        self.add_n_spectra(1)
        self.ds.add_peak(120, fitting_type="gauss")
        self.ds.add_peak(100, fitting_type="lorentz")
        params = self.prefitter._generate_params_list()
        self.assertListEqual(
            list(params.keys()),
            [
                "Peak_at_150_ppm_amp_0",
                "Peak_at_150_ppm_cen_0",
                "Peak_at_150_ppm_sigma_0",
                "Peak_at_150_ppm_gamma_0",
                "Peak_at_120_ppm_amp_0",
                "Peak_at_120_ppm_cen_0",
                "Peak_at_120_ppm_sigma_0",
                "Peak_at_100_ppm_amp_0",
                "Peak_at_100_ppm_cen_0",
                "Peak_at_100_ppm_gamma_0",
            ],
        )

    def test_sort_params_one_voigt_one_spectrum(self):
        self.add_n_spectra(1)
        params = self.prefitter._generate_params_list()
        param_dict_list = self.prefitter._sort_params(params)
        self.assertDictEqual(
            param_dict_list, {0: [[200.0, 150.0, 10.0, 10.0, "gam"]]}
        )

    def test_sort_params_two_voigt_one_spectrum_user_def_sigma_max(self):
        self.add_n_spectra(1)
        self.ds.add_peak(120, line_broadening={"sigma": {"max": 2}})
        params = self.prefitter._generate_params_list()
        param_dict_list = self.prefitter._sort_params(params)
        self.assertDictEqual(
            param_dict_list,
            {
                0: [
                    [200.0, 150.0, 10.0, 10.0, "gam"],
                    [200.0, 120.0, 1.0, 10.0, "gam"],
                ]
            },
        )

    def test_sort_params_voigt_gauss_lorentz_one_spectrum(self):
        self.add_n_spectra(1)
        self.ds.add_peak(120, fitting_type="gauss")
        self.ds.add_peak(100, fitting_type="lorentz")
        params = self.prefitter._generate_params_list()
        param_dict_list = self.prefitter._sort_params(params)
        self.assertDictEqual(
            param_dict_list,
            {
                0: [
                    [200.0, 150.0, 10.0, 10.0, "gam"],
                    [200.0, 120.0, 10.0],
                    [200.0, 100.0, 10.0, "gam"],
                ]
            },
        )

    def test_spectral_fitting_one_voigt_without_noise_prefit(self):
        self.add_n_spectra(1)
        x_axis, y_axis = self.prefitter._generate_axis_list()
        params = self.prefitter._generate_params_list()
        params["Peak_at_150_ppm_cen_0"].max = 250
        params["Peak_at_150_ppm_cen_0"].value = 250
        params["Peak_at_150_ppm_sigma_0"].value = 2
        params["Peak_at_150_ppm_gamma_0"].value = 2
        residual = self.prefitter._spectral_fitting(params, x_axis, y_axis)
        self.assertEqual(sum(residual), 0)

    def test_spectral_fitting_one_gauss_prefit(self):
        self.add_n_spectra(1, type=["gauss"])
        self.ds.peak_list[0].fitting_type = "gauss"
        x_axis, y_axis = self.prefitter._generate_axis_list()
        params = self.prefitter._generate_params_list()
        params["Peak_at_150_ppm_sigma_0"].value = 3
        residual = self.prefitter._spectral_fitting(params, x_axis, y_axis)
        self.assertEqual(sum(residual), 0)

    def test_spectral_fitting_one_lorentz_prefit(self):
        self.add_n_spectra(1, type=["lorentz"])
        self.ds.peak_list[0].fitting_type = "lorentz"
        x_axis, y_axis = self.prefitter._generate_axis_list()
        params = self.prefitter._generate_params_list()
        params["Peak_at_150_ppm_gamma_0"].value = 4
        params["Peak_at_150_ppm_cen_0"].max = 200
        params["Peak_at_150_ppm_cen_0"].value = 200
        residual = self.prefitter._spectral_fitting(params, x_axis, y_axis)
        self.assertEqual(sum(residual), 0)

    def test_spectral_fitting_voigt_gauss_lorentz_prefit(self):
        self.add_n_spectra(1, type=["voigt", "gauss", "lorentz"])
        self.ds.add_peak(200, fitting_type="gauss")
        self.ds.add_peak(250, fitting_type="lorentz")
        x_axis, y_axis = self.prefitter._generate_axis_list()
        params = self.prefitter._generate_params_list()
        params["Peak_at_150_ppm_cen_0"].max = 250
        params["Peak_at_150_ppm_cen_0"].value = 250
        params["Peak_at_200_ppm_cen_0"].min = 150
        params["Peak_at_200_ppm_cen_0"].value = 150
        params["Peak_at_250_ppm_cen_0"].min = 200
        params["Peak_at_250_ppm_cen_0"].value = 200
        params["Peak_at_150_ppm_sigma_0"].value = 2
        params["Peak_at_150_ppm_gamma_0"].value = 2
        params["Peak_at_200_ppm_sigma_0"].value = 3
        params["Peak_at_250_ppm_gamma_0"].value = 4
        residual = self.prefitter._spectral_fitting(params, x_axis, y_axis)
        self.assertAlmostEqual(sum(residual), 0, delta=1e-5)

    def test_prefiter_fit_one_voigt(self):
        self.add_n_spectra(1)
        self.prefitter.dataset.peak_list[0].peak_center = 250
        result = self.prefitter.fit()
        value_list = []
        for key in result.params:
            value_list.append(round(result.params[key].value))
        self.assertListEqual(value_list, [200, 250, 2, 2])

    def test_prefiter_fit_voigt_gauss(self):
        self.add_n_spectra(1, type=["voigt", "gauss"])
        self.ds.add_peak(210, fitting_type="gauss")
        self.prefitter.dataset.peak_list[0].peak_center = 250
        self.prefitter.dataset.peak_list[1].peak_center = 150
        result = self.prefitter.fit()
        value_list = []
        for key in result.params:
            value_list.append(round(result.params[key].value))
        self.assertListEqual(value_list, [200, 250, 2, 2, 200, 150, 3])

    def test_prefiter_fit_voigt_gauss(self):
        self.add_n_spectra(1, type=["voigt", "gauss", "lorentz"])
        self.ds.add_peak(210, fitting_type="gauss")
        self.ds.add_peak(200, fitting_type="lorentz")
        self.prefitter.dataset.peak_list[0].peak_center = 250
        self.prefitter.dataset.peak_list[1].peak_center = 150
        result = self.prefitter.fit()
        value_list = []
        for key in result.params:
            value_list.append(round(result.params[key].value))
        self.assertListEqual(
            value_list, [200, 250, 2, 2, 200, 150, 3, 200, 200, 4]
        )

    def test_prefiter_fit_one_voigt_add_noise_test_amop_cen(self):
        self.add_n_spectra(1)
        self.ds.spectra[0].y_axis = self.add_noise(self.ds.spectra[0].y_axis)
        self.prefitter.dataset.peak_list[0].peak_center = 250
        result = self.prefitter.fit()
        value_list = []
        for key in result.params:
            value_list.append(round(result.params[key].value))
        self.assertListAlmostEqual(value_list[:2], [200, 250], delta=5)

    def test_prefiter_fit_one_voigt_add_noise_test_lw(self):
        self.add_n_spectra(1)
        self.ds.spectra[0].y_axis = self.add_noise(self.ds.spectra[0].y_axis)
        self.prefitter.dataset.peak_list[0].peak_center = 250
        result = self.prefitter.fit()
        value_list = []
        for key in result.params:
            value_list.append(round(result.params[key].value))
        self.assertListAlmostEqual(value_list[2:], [2, 2], delta=0.01)

    def test_prefiter_fit_voigt_gauss_with_noise(self):
        self.add_n_spectra(1, type=["voigt", "gauss", "lorentz"])
        self.ds.add_peak(210, fitting_type="gauss")
        self.ds.add_peak(200, fitting_type="lorentz")
        self.ds.spectra[0].y_axis = self.add_noise(self.ds.spectra[0].y_axis)
        self.prefitter.dataset.peak_list[0].peak_center = 250
        self.prefitter.dataset.peak_list[1].peak_center = 150
        result = self.prefitter.fit()
        value_list = []
        for key in result.params:
            value_list.append(round(result.params[key].value))
        self.assertListAlmostEqual(
            value_list, [200, 250, 2, 2, 200, 150, 3, 200, 200, 4], delta=8
        )

    def test_generate_axis_list_global_fitter_nr_elements_x(self):
        self.add_n_spectra(3)
        x_axis, y_axis = self.globalfitter._generate_axis_list()
        self.assertEqual(len(x_axis), 3)

    def test_generate_axis_list_global_fitter_nr_elements_y(self):
        self.add_n_spectra(3)
        x_axis, y_axis = self.globalfitter._generate_axis_list()
        self.assertEqual(len(y_axis), 3)

    def test_generate_axis_list_global_fitter_corect_x_val(self):
        self.add_n_spectra(3)
        x_axis, y_axis = self.globalfitter._generate_axis_list()
        self.assertTrue(
            np.array_equal(x_axis[1], np.linspace(100, 350, 1000))
        )

    def test_generate_axis_list_global_fitter_corect_y_val(self):
        self.add_n_spectra(3)
        x_axis, y_axis = self.globalfitter._generate_axis_list()

        max_list = []
        for vals in y_axis:
            plt.plot(x_axis[0], vals)
            max_list.append(int(max(vals)))

        self.assertListEqual(max_list, [20, 41, 62])

    def test_generate_axis_list_single_fitter_nr_elements_x(self):
        self.add_n_spectra(3)
        x_axis, y_axis = self.singlefitter._generate_axis_list()
        self.assertEqual(len(x_axis), 3)

    def test_generate_axis_list_single_fitter_nr_elements_y(self):
        self.add_n_spectra(3)
        x_axis, y_axis = self.singlefitter._generate_axis_list()
        self.assertEqual(len(y_axis), 3)

    def test_generate_axis_list_single_fitter_corect_x_val(self):
        self.add_n_spectra(3)
        x_axis, y_axis = self.singlefitter._generate_axis_list()
        self.assertTrue(
            np.array_equal(x_axis[1], np.linspace(100, 350, 1000))
        )

    def test_generate_axis_list_single_fitter_corect_y_val(self):
        self.add_n_spectra(3)
        x_axis, y_axis = self.globalfitter._generate_axis_list()
        max_list = []
        for vals in y_axis:
            plt.plot(x_axis[0], vals)
            max_list.append(int(max(vals)))
        self.assertListEqual(max_list, [20, 41, 62])

    def test_generate_param_list_two_spectra_global_fitter(self):
        self.add_n_spectra(2)
        params = self.globalfitter._generate_params_list()
        keylist = []
        for keys in params.keys():
            keylist.append(keys)
        self.assertListEqual(
            keylist,
            [
                "Peak_at_150_ppm_amp_0",
                "Peak_at_150_ppm_cen_0",
                "Peak_at_150_ppm_sigma_0",
                "Peak_at_150_ppm_gamma_0",
                "Peak_at_150_ppm_amp_1",
                "Peak_at_150_ppm_cen_1",
                "Peak_at_150_ppm_sigma_1",
                "Peak_at_150_ppm_gamma_1",
            ],
        )

    def test_generate_param_list_two_spectra_single_fitter(self):
        self.add_n_spectra(2)
        params = self.singlefitter._generate_params_list()
        keylist = []
        for keys in params.keys():
            keylist.append(keys)
        self.assertListEqual(
            keylist,
            [
                "Peak_at_150_ppm_amp_0",
                "Peak_at_150_ppm_cen_0",
                "Peak_at_150_ppm_sigma_0",
                "Peak_at_150_ppm_gamma_0",
                "Peak_at_150_ppm_amp_1",
                "Peak_at_150_ppm_cen_1",
                "Peak_at_150_ppm_sigma_1",
                "Peak_at_150_ppm_gamma_1",
            ],
        )
