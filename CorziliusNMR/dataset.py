import lmfit
from CorziliusNMR import io, utils, settings
import numpy as np
import sys


class Dataset:

    def __init__(self):
        self.importer = None
        self.props = settings.Properties()
        self.spectra = []
        self.fitter = None
        self.peak_list = []

    def start_buildup_fit_from_topspin_export(self):  # TODO Test
        print(
            f"Start loading data from topspin: {self.path_to_topspin_experiment}"
        )
        self._read_in_data_from_topspin()
        print("Start peak fitting.")
        self._calculate_peak_intensities()
        print("Start buildup fit.")
        self._buidup_fit_global()
        self._print()

    def start_buildup_fit_from_spectra(self):
        print(
            "This function is work in progress and can't be used. Program will exit."
        )
        sys.exit()
        self._read_in_data_from_csv()
        self._calculate_peak_intensities()
        self._buidup_fit_global()

    def start_buildup_from_intensitys(self):
        print(
            "This function is work in progress and can't be used. Program will exit."
        )
        sys.exit()
        return

    def add_peak(
        self,
        center_of_peak,
        peak_label="",
        fitting_group=1,
        fitting_type="voigt",
        peak_sign="+",
    ):
        self.peak_list.append(Peak())
        self.peak_list[-1].peak_center = center_of_peak
        self.peak_list[-1].peak_label = peak_label
        self.peak_list[-1].fitting_group = fitting_group
        self.peak_list[-1].fitting_type = fitting_type
        self.peak_list[-1].peak_sign = peak_sign

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
        pass

    def _read_in_data_from_csv(self):
        # TODO
        pass

    def _calculate_peak_intensities(self):
        self._add_peaks_to_all_exp()
        if "fit" in self.spectrum_fitting_type:
            self._perform_spectrum_fit()
        if "global" in self.spectrum_fitting_type:
            self._perform_global_spectrum_fit()

    def _buidup_fit_global(self):
        for type in self.props.buildup_types:
            if type == "biexponential":
                buildup_fitter = utils.BiexpFitter(self)
                buildup_fitter.perform_fit()
            elif type == "biexponential_with_offset":
                buildup_fitter = utils.BiexpFitterWithOffset(self)
                buildup_fitter.perform_fit()
            elif type == "exponential":
                buildup_fitter = utils.ExpFitter(self)
                buildup_fitter.perform_fit()
            elif type == "exponential_with_offset":
                buildup_fitter = utils.ExpFitterWithOffset(self)
                buildup_fitter.perform_fit()

    def _add_peaks_to_all_exp(self):
        for spectrum in self.spectra:
            spectrum.add_peak(self.peak_dict)
            for peak in spectrum.peaks:
                peak.assign_values_from_dict()

    def _perform_spectrum_fit(self):
        pass

    def _perform_global_spectrum_fit(self):
        self.fitter = utils.GlobalSpectrumFitter(self)
        self.fitter.start_prefit()
        self.fitter.set_model()
        self.fitter.fit()

    def _get_intensities(self):
        pass


class Spectra:

    def __init__(self):
        self.number_of_scans = None
        self.tdel = None
        self.x_axis = None
        self.y_axis = None
        self.peaks = None


class Peak:

    def __init__(self):
        self._peak_center = None
        self._peak_label = None
        self._fitting_group = None
        self._fitting_type = None
        self._peak_sign = None
        self._line_broadening = None  # TODO

    @property
    def line_broadening(self) -> str:
        return self._line_broadening

    @line_broadening.setter
    def line_broadening(self, value):
        allowed_values = ["sigma", "gamma"]
        inner_allowed_values = ["min", "max"]
        if not isinstance(value, dict):
            raise TypeError(
                f"'line_broadening' must be of type 'dict', but got {type(value)}."
            )
        if not all(key in allowed_values for key in value.keys()):
            raise ValueError(
                f"One or more keys in the dictionary are not in the allowed values: {allowed_values}!"
            )
        if not all(isinstance(values, dict) for values in value.values()):
            raise TypeError(
                f"Each value in the 'line_broadening' dictionary must be of type 'dict'."
            )
        for keys in value.keys():
            if not all(
                key in inner_allowed_values for key in value[keys].keys()
            ):
                raise ValueError(
                    f"One or more keys in the dictionary are not in the allowed values: {inner_allowed_values}."
                )
        params = self._return_default_dict()
        if self.fitting_type == "gauss":
            params = {"sigma": params["sigma"]}
        if self.fitting_type == "lorentz":
            params = {"gamma": params["gamma"]}
        for key in allowed_values:
            if key in value:
                for inner_key in inner_allowed_values:
                    if inner_key in value[key]:
                        if not isinstance(
                            value[key][inner_key], (int, float)
                        ):
                            raise TypeError(
                                f"Each value in dicts must be 'int' or 'float', but found {type(value[key][inner_key])}."
                            )
                        params[key][inner_key] = float(value[key][inner_key])
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
        if not value in allowed_values:
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
        if not value in allowed_values:
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
