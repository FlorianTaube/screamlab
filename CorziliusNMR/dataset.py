import lmfit
from CorziliusNMR import io, utils, settings
import numpy as np


class Dataset:

    def __init__(self):
        self.importer = None
        self.props = settings.Properties()
        self.spectra = []
        self.fitter = None
        self.peak_list = []
        self.lmfit_result_handler = io.LmfitResultHandler()

    def start_buildup_fit_from_topspin_export(self):  # TODO Test
        print(
            f"Start loading data from topspin: {self.path_to_topspin_experiment}"
        )
        self._read_in_data_from_topspin()
        print("Start peak fitting.")
        self._calculate_peak_intensities()
        print("Start buildup fit.")
        self._start_buildup_fit()
        self._print()

    def start_buildup_fit_from_spectra(self):
        self._read_in_data_from_csv()
        self._calculate_peak_intensities()
        self._start_buildup_fit()

    def start_buildup_from_intensitys(self):
        return

    def add_peak(
        self,
        center_of_peak,
        peak_label="",
        fitting_group=1,
        fitting_type="voigt",
        peak_sign="+",
        line_broadening=dict(),
    ):
        self.peak_list.append(Peak())
        self.peak_list[-1].peak_center = center_of_peak
        self.peak_list[-1].peak_label = peak_label
        self.peak_list[-1].fitting_group = fitting_group
        self.peak_list[-1].fitting_type = fitting_type
        self.peak_list[-1].peak_sign = peak_sign
        self.peak_list[-1].line_broadening = line_broadening

    def _read_in_data_from_topspin(self):  # TODO test
        self._setup_correct_topspin_importer()
        self.importer.import_topspin_data()

    def _setup_correct_topspin_importer(self):
        if len(self.props.expno) == 1:
            self.importer = io.Pseudo2DImporter(self)
        else:
            self.importer = io.ScreamImporter(self)

    def _print(self):
        exporter = io.Exporter(self)
        exporter.print_all()

    def _read_in_data_from_csv(self):
        # TODO
        pass

    def _calculate_peak_intensities(self):
        if self.props.prefit:
            self._set_prefitter()
            result = self.fitter.fit()
            self.lmfit_result_handler.prefit = lmfit.fit_report(result)
            self._update_line_broadening(result)
        if "individual" in self.props.spectrum_fit_type:
            self._set_single_fitter()
            result = self.fitter.fit()
            self.lmfit_result_handler.single_fit = lmfit.fit_report(
                result, self.spectra
            )
            self._get_intensities(result)
        if "global" in self.props.spectrum_fit_type:
            self._set_global_fitter()
            result = self.fitter.fit()
            self.lmfit_result_handler.global_fit = lmfit.fit_report(result)
            self._get_intensities(result)

    def _start_buildup_fit(self):
        fitter_classes = {
            "biexponential": utils.BiexpFitter,
            "biexponential_with_offset": utils.BiexpFitterWithOffset,
            "exponential": utils.ExpFitter,
            "exponential_with_offset": utils.ExpFitterWithOffset,
        }

        for b_type in self.props.buildup_types:
            fitter_class = fitter_classes.get(b_type)
            if fitter_class:
                fitter = fitter_class(self)
                result = fitter.perform_fit()

    def _set_prefitter(self):
        self.fitter = utils.Prefitter(self)

    def _set_single_fitter(self):
        self.fitter = utils.SingleFitter(self)

    def _set_global_fitter(self):
        self.fitter = utils.GlobalFitter(self)

    def _get_intensities(self, result):
        if isinstance(self.fitter, utils.SingleFitter):
            for peak in self.peak_list:
                peak.individual_fit_vals = (result, self.spectra)
        if isinstance(self.fitter, utils.GlobalFitter):
            for peak in self.peak_list:
                peak.global_fit_vals = (result, self.spectra)

    def _update_line_broadening(self, result):
        for peak in self.peak_list:
            value = dict()
            for lw in ["sigma", "gamma"]:
                key = (
                    f"{peak.peak_label}_{lw}_{self.props.spectrum_for_prefit}"
                )
                if key in result.params.keys():
                    value.update(
                        {
                            f"{lw}": {
                                "min": round(
                                    result.params[key].value
                                    - 0.1 * result.params[key].value,
                                    2,
                                ),
                                "max": round(
                                    result.params[key].value
                                    + 0.1 * result.params[key].value,
                                    2,
                                ),
                            }
                        }
                    )
            peak.line_broadening = value


class Spectra:

    def __init__(self):
        self.number_of_scans = None
        self.tdel = None
        self.x_axis = None
        self.y_axis = None


class Peak:

    def __init__(self):
        self._peak_center = None
        self._peak_label = None
        self._fitting_group = None
        self._fitting_type = None
        self._peak_sign = None
        self._line_broadening = None
        self._individual_fit_vals = None
        self._global_fit_vals = None

    @property
    def individual_fit_vals(self) -> list:
        return self._individual_fit_vals

    @individual_fit_vals.setter
    def individual_fit_vals(self, args):
        result, spectra = args
        self._individual_fit_vals = BuildupList()
        self._individual_fit_vals.set_vals(result, spectra, self.peak_label)

    @property
    def global_fit_vals(self) -> list:
        return self._global_fit_vals

    @global_fit_vals.setter
    def global_fit_vals(self, args):
        result, spectra = args
        self._global_fit_vals = BuildupList()
        self._global_fit_vals.set_vals(result, spectra, self.peak_label)

    @property
    def line_broadening(self) -> str:
        return self._line_broadening

    @line_broadening.setter
    def line_broadening(self, value):
        allowed_values = ["sigma", "gamma"]
        inner_allowed_values = ["min", "max"]
        self._check_if_value_is_dict(value)
        self._check_for_invalid_keys(value, allowed_values)
        self._check_for_invalid_dict(value)
        self._check_for_invalid_inner_keys(value, inner_allowed_values)
        params = self._set_init_params()
        self._overwrite_init_params(
            value, allowed_values, inner_allowed_values, params
        )
        self._line_broadening = params

    @property
    def peak_sign(self) -> str:
        return self._peak_sign

    @peak_sign.setter
    def peak_sign(self, value):
        allowed_values = {"+", "-"}
        if not isinstance(value, str):
            raise TypeError(
                f"'peak_sign' must be of type 'str', but got {type(value)}."
            )
        if value not in allowed_values:
            raise ValueError(
                f"All elements in 'peak_sign' must be one of {sorted(allowed_values)}."
            )
        self._peak_sign = value

    @property
    def fitting_type(self) -> str:
        return self._fitting_type

    @fitting_type.setter
    def fitting_type(self, value):
        allowed_values = {
            "voigt",
            "lorentz",
            "gauss",
        }
        if not isinstance(value, str):
            raise TypeError(
                f"'fitting_type' must be of type 'str', but got {type(value)}."
            )
        if value not in allowed_values:
            raise ValueError(
                f"All elements in 'fitting_type' must be one of {sorted(allowed_values)}."
            )
        self._fitting_type = value

    @property
    def fitting_group(self) -> int:
        return self._fitting_group

    @fitting_group.setter
    def fitting_group(self, value):
        if not isinstance(value, int):
            raise TypeError(
                f"'fitting_group' must be of type 'int', but got {type(value)}."
            )
        self._fitting_group = value

    @property
    def peak_center(self) -> (int, float):
        return self._peak_center

    @peak_center.setter
    def peak_center(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"'peak_center' must be of type 'int' or 'float', but got {type(value)}."
            )
        self._peak_center = float(value)

    @property
    def peak_label(self) -> str:
        return self._peak_label

    @peak_label.setter
    def peak_label(self, value):
        if not isinstance(value, str):
            raise TypeError(
                f"'peak_label' must be of type 'str', but got {type(value)}."
            )
        if value == "":
            value = f"Peak_at_{int(self.peak_center)}_ppm"
        self._peak_label = value

    def _return_default_dict(self):
        return {
            "sigma": {"min": 0, "max": 20},
            "gamma": {"min": 0, "max": 20},
        }

    def _check_if_value_is_dict(self, value):
        if not isinstance(value, dict):
            raise TypeError(
                f"'line_broadening' must be a 'dict', but got {type(value)}."
            )

    def _check_for_invalid_keys(self, value, allowed_values):
        invalid_keys = [
            key for key in value.keys() if key not in allowed_values
        ]
        if invalid_keys:
            raise ValueError(
                f"Invalid keys found in the dictionary: {invalid_keys}. Allowed keys are: {allowed_values}."
            )

    def _check_for_invalid_dict(self, value):
        if not all(isinstance(v, dict) for v in value.values()):
            raise TypeError(
                "Each value in the 'line_broadening' dictionary must be of type 'dict'."
            )

    def _check_for_invalid_inner_keys(self, value, inner_allowed_values):
        for key, inner_dict in value.items():
            invalid_inner_keys = [
                inner_key
                for inner_key in inner_dict.keys()
                if inner_key not in inner_allowed_values
            ]
            if invalid_inner_keys:
                raise ValueError(
                    f"Invalid inner keys for '{key}': {invalid_inner_keys}. Allowed inner keys are: {inner_allowed_values}."
                )

    def _set_init_params(self):
        params = self._return_default_dict()
        if self.fitting_type == "gauss":
            params = {"sigma": params["sigma"]}
        elif self.fitting_type == "lorentz":
            params = {"gamma": params["gamma"]}
        return params

    def _overwrite_init_params(
        self, value, allowed_values, inner_allowed_values, params
    ):
        for key in allowed_values:
            if key in value:
                for inner_key in inner_allowed_values:
                    inner_value = value[key].get(inner_key)
                    if inner_value is not None:
                        if not isinstance(inner_value, (int, float)):
                            raise TypeError(
                                f"'{inner_key}' value must be an 'int' or 'float', but got {type(inner_value)}."
                            )
                        params[key][inner_key] = float(inner_value)
        return params


class BuildupList:

    def __init__(self):
        self._tdel = None
        self._intensity = None

    def set_vals(self, result, spectra, label):
        self._set_tdel(spectra)
        self._set_intensity(result, label, spectra)
        self._sort_lists()

    def _set_tdel(self, spectra):
        self._tdel = [s.tdel for s in spectra]

    def _set_intensity(self, result, label, spectra):
        last_digid = None
        self._intensity = []
        val_list = []
        for param in result.params:
            if label in param:
                if last_digid != param.split("_")[-1]:
                    if val_list != []:
                        self._intensity.append(
                            self._calc_integral(
                                val_list, spectra[int(last_digid)]
                            )
                        )
                    last_digid = param.split("_")[-1]
                    val_list = []
                val_list.append(float(result.params[param].value))
                if param.split("_")[-2] == "gamma":
                    val_list.append("gamma")
        self._intensity.append(
            self._calc_integral(val_list, spectra[int(last_digid)])
        )

    def _calc_integral(self, val_list, spectrum):
        if len(val_list) == 5:
            sim_spectrum = utils.voigt_profile(
                spectrum.x_axis,
                val_list[1],
                val_list[2],
                val_list[3],
                val_list[0],
            )
        if len(val_list) == 4:
            sim_spectrum = utils.lorentz_profile(
                spectrum.x_axis, val_list[1], val_list[2], val_list[0]
            )
        if len(val_list) == 3:
            sim_spectrum = utils.gauss_profile(
                spectrum.x_axis, val_list[1], val_list[2], val_list[0]
            )
        return np.trapz(sim_spectrum)

    def _sort_lists(self):
        self._tdel, self._intensity = map(
            list, zip(*sorted(zip(self._tdel, self._intensity)))
        )
