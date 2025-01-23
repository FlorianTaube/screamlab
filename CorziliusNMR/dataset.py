import lmfit
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


class Peaks:

    def __init__(self):
        self._peak_list = []
        lmfit.Parameters
