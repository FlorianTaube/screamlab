from CorziliusNMR import io,plotter,utils
import numpy as np
import sys
from bruker.data.nmr import *
import bruker.api.topspin as top


class Dataset:

    def __init__(self):
        self._fileNames = io._FileNameHandler()
        self._topspin_exporter = None
        self.experiments = []
        self.peak_dict = None
        self.spectrum_fitting_type = ["max_value","fit","global_fit"]

    @property
    def path_to_topspin_experiment(self):
        return self._fileNames.path_to_topspin_experiment

    @path_to_topspin_experiment.setter
    def path_to_topspin_experiment(self, path):
        self._fileNames.path_to_topspin_experiment = path

    @property
    def output_file_name(self):
        return self._fileNames.output_file_name

    @output_file_name.setter
    def output_file_name(self, file):
        self._fileNames.output_file_name = file

    @property
    def procnoOfTopspinExperiment(self):
        return self._fileNames.procno_of_topspin_experiment

    @procnoOfTopspinExperiment.setter
    def procnoOfTopspinExperiment(self, procno):
        self._fileNames.procno_of_topspin_experiment = str(procno)

    @property
    def expno_of_topspin_experiment(self):
        return self._fileNames.expno_of_topspin_experiment

    @expno_of_topspin_experiment.setter
    def expno_of_topspin_experiment(self, expno):
        if type(expno) != list:
            sys.exit("Wrong format. List expected.")
        if len(expno) == 2:
            self._fileNames.expno_of_topspin_experiment = np.arange(expno[0], expno[1] + 1)
        else:
            self._fileNames.expno_of_topspin_experiment = expno


    def start_buildup_fit_from_topspin_export(self):
        self._read_in_data_from_topspin()
        self._calculate_peak_intensities()
        #self._


    def start_buildup_fit_from_spectra(self):
        self._read_in_data_from_csv()
        self._calculate_peak_intensities()
        return

    def start_buildup_from_intensitys(self):
        return

    def _setup_correct_topspin_exporter(self):
        if len(self.expno_of_topspin_experiment) == 1:
            self._topspin_exporter = io.Pseudo2DExporter(self)
        else:
            self._topspin_exporter = io.ScreamExporter(self)

    def _read_in_data_from_topspin(self):
        self._setup_correct_topspin_exporter()
        self._topspin_exporter.export()
        _plotter = plotter.ExperimentalSpectraPlotter(self)
        _plotter.plot()

    def _read_in_data_from_csv(self):
        #TODO
        pass

    def _calculate_peak_intensities(self):
        self._add_peaks_to_all_exp()
        if "fit" in self.spectrum_fitting_type:
            self._perform_spectrum_fit()
        if "global_fit" in self.spectrum_fitting_type:
            self._perform_global_spectrum_fit()



    def _add_peaks_to_all_exp(self):
        for spectrum in self.experiments:
            spectrum.add_peak(self.peak_dict)

    def _perform_spectrum_fit(self):
        pass

    def _perform_global_spectrum_fit(self):
        pass

    def _get_intensities(self):
        pass








class _Experiment():

    def __init__(self,file):
        self._file = file
        self.NS = ""
        self.tbup = ""
        self.x_axis = None
        self.y_axis = []
        self._top = top.Topspin()
        self._data_provider = self._top.getDataProvider()
        self._nmr_metadata = self._data_provider.getNMRData(self._file)
        self._nmr_spectral_data = self._nmr_metadata.getSpecDataPoints()
        self.peaks = []

    def _get_values(self):
        self._get_number_of_scans()
        self._get_buildup_time()
        self._get_x_axis()
        self._get_y_axis()
        self._normalize_y_values_to_number_of_scans()

    def _get_number_of_scans(self):
        self.NS = int(self._nmr_metadata.getPar("NS"))

    def _get_buildup_time(self):
        self.tbup = int(self._nmr_metadata.getPar("L 20")) / 4

    def _get_x_axis(self):
        _physicalRange = self._nmr_spectral_data['physicalRanges'][0]
        _number_of_datapoints = self._nmr_spectral_data['dataPoints']
        self.x_axis = np.linspace(float(_physicalRange['start']),
                                  float(_physicalRange[ 'end']),
                                  len(_number_of_datapoints))

    def _get_y_axis(self):
        self.y_axis = self._nmr_spectral_data['dataPoints']

    def _normalize_y_values_to_number_of_scans(self):
        self.y_axis = np.divide(self.y_axis, self.NS)

    def add_peak(self,peak_dict):
        try:
            for peak in sorted(peak_dict.keys(), key=lambda x: int(x),
                           reverse=True):
                self.peaks.append(_Peak(peak,peak_dict[peak],self))
        except:
            print("ERROR: No peaks given. Try dataset.something")#TODO
            # Write correct function as information
            pass

class _Peak():

    def __init__(self,peak,peak_dict,experminent):
        self.sign = None
        self.peak_center_rounded = int(peak)
        self.peak_label = None
        self.hight = None
        self.fwhm = None
        self.area_under_peak = None
        self.fitting_group = None
        self.peak_dict = peak_dict
        self.experiment = experminent
        self.fitting_model = None
        self._assign_values_from_dict()

    def _assign_values_from_dict(self):
        self._set_sign()
        self._set_peak_label()
        self._set_fitting_group()
        self._set_fitting_model()
        self._set_hight()

    def _set_sign(self):
        try:
            if self.peak_dict['sign'] in ["+","-"]:
                self.sign = self.peak_dict['sign']
            else:
                print("ERROR: Wrong input for peak sign. Must be \"+\" or "
                      "\"-\". Set peak sign to default value.")
                self.sign = "+"
        except:
            self.sign = "+"

    def _set_peak_label(self):
        try:
            self.peak_label = self.peak_dict['label']
        except:
            self.peak_label = f"Peak_at_{self.peak_center_rounded}_ppm"

    def _set_fitting_group(self):
        try:
            self.fitting_group = int(self.peak_dict['fitting_group'])
        except:
            self.fitting_group = 999

    def _set_fitting_model(self):
        try:
            if self.peak_dict['fitting_model'] in ["voigt","gauss","lorentz"]:
                self.fitting_model = peak_dict['fitting_model']
            else:
                print(f"ERROR: Unknown fitting model: "
                      f"{self.peak_dict['fitting_model']}. \'voigt\',"
                      f"\'gauss\' or \'lorentz\' expected. Set fitting_model "
                      f"to "
                      f"default.")
                self.fitting_model = "voigt"
        except:
            self.fitting_model = "voigt"

    def _set_hight(self):
        subspectrum = utils.generate_subspectrum(self.experiment,
                                                 self.peak_center_rounded, 4)
        if np.trapz(subspectrum) < 0:
            y_val = min(subspectrum)
        else:
            y_val = max(subspectrum)
        index = np.where(self.experiment.y_axis == y_val)
        x_val = self.experiment.x_axis[index]
        self.hight = {'index':index[0],'x_val':x_val[0],'y_val':y_val}