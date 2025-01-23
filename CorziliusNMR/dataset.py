from CorziliusNMR import io, utils, settings
import numpy as np


class Dataset:

    def __init__(self):
        self.importer = None
        self.props = settings.Properties()
        self.spectra = []
        self.fitter = None

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
        self._read_in_data_from_csv()
        self._calculate_peak_intensities()
        self._buidup_fit_global()

    def start_buildup_from_intensitys(self):
        return

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

    def __init__(self, spectrum, peak, peak_dict):
        self.peak_center_rounded = int(peak)
        self.spectrum = spectrum
        self.peak_dict = peak_dict[peak]
        self.sign = None
        self.peak_label = None
        self.hight = None
        self.fwhm = None
        self.area_under_peak = dict()
        self.simulated_peak = dict()
        self.fitting_parameter = dict()
        self.fitting_report = dict()
        self.fitting_group = None
        self.fitting_model = None
        self.prefit_dict = None

    def assign_values_from_dict(self):
        self._set_sign()
        self._set_peak_label()
        self._set_fitting_group()
        self._set_fitting_model()
        self._set_hight()

    def _set_sign(self):
        try:
            if self.peak_dict["sign"] in ["+", "-"]:
                self.sign = self.peak_dict["sign"]
            else:
                print(
                    'ERROR: Wrong input for peak sign. Must be "+" or '
                    '"-". Set peak sign to default value.'
                )
                self.sign = "+"
        except:
            self.sign = "+"

    def _set_peak_label(self):
        try:
            self.peak_label = self.peak_dict["label"]
        except:
            self.peak_label = (
                f"Peak_at_{self.peak_center_rounded}_ppm_"
                f"{self.spectrum.tbup}_s"
            )
            self.peak_label = self.peak_label.replace("-", "m")

    def _set_fitting_group(self):
        try:
            self.fitting_group = int(self.peak_dict["fitting_group"])
        except:
            self.fitting_group = 999

    def _set_fitting_model(self):
        try:
            if self.peak_dict["fitting_model"] in [
                "voigt",
                "gauss",
                "lorentz",
            ]:
                self.fitting_model = self.peak_dict["fitting_model"]
            else:
                print(
                    f"ERROR: Unknown fitting model: "
                    f"{self.peak_dict['fitting_model']}. 'voigt',"
                    f"'gauss' or 'lorentz' expected. Set fitting_model "
                    f"to "
                    f"default."
                )
                self.fitting_model = "voigt"
        except:
            self.fitting_model = "voigt"

    def _set_hight(self):
        subspectrum = utils.generate_subspectrum(
            self.spectrum, self.peak_center_rounded, 2
        )

        if np.trapz(subspectrum) < 0:
            y_val = min(subspectrum)
        else:
            y_val = max(subspectrum)
        index = np.where(self.spectrum.y_axis == y_val)[0]
        x_val = self.spectrum.x_axis[index]
        self.hight = {"index": index[0], "x_val": x_val[0], "y_val": y_val}
