from CorziliusNMR import io,utils
import numpy as np
import sys
from bruker.data.nmr import *
import bruker.api.topspin as top
import os


class Dataset:

    def __init__(self):
        self.file_name_generator = FileNameHandler()
        self.importer = None
        self.spectra = []
        self.peak_dict = dict()
        self.spectrum_fitting_type = ["global"]
        self.fitter = None
        self.buildup_type = ["biexponential"]

    @property
    def path_to_topspin_experiment(self):
        return self.file_name_generator.path_to_topspin_experiment

    @path_to_topspin_experiment.setter
    def path_to_topspin_experiment(self, path):
        self.file_name_generator.path_to_topspin_experiment = path

    @property
    def output_file_name(self):
        return self.file_name_generator.output_file_name

    @output_file_name.setter
    def output_file_name(self, file):
        self.file_name_generator.output_file_name = file

    @property
    def procno_of_topspin_experiment(self):
        return self.file_name_generator.procno_of_topspin_experiment

    @procno_of_topspin_experiment.setter
    def procno_of_topspin_experiment(self, procno):
        self.file_name_generator.procno_of_topspin_experiment = str(procno)

    @property
    def expno_of_topspin_experiment(self):
        return self.file_name_generator.expno_of_topspin_experiment

    @expno_of_topspin_experiment.setter
    def expno_of_topspin_experiment(self, expno):
        if type(expno) != list:
            sys.exit("Wrong format. List expected.")
        if len(expno) == 2:
            self.file_name_generator.expno_of_topspin_experiment = np.arange(expno[0], expno[1] + 1)
        else:
            self.file_name_generator.expno_of_topspin_experiment = expno


    def start_buildup_fit_from_topspin_export(self):
        self._read_in_data_from_topspin()
        self._calculate_peak_intensities()
        self._buidup_fit_global()
        self._print()


    def _print(self):
        exporter = io.Exporter(self)
        exporter.print_all()
        pass

    def start_buildup_fit_from_spectra(self):
        self._read_in_data_from_csv()
        self._calculate_peak_intensities()
        self._buidup_fit_global()

    def start_buildup_from_intensitys(self):
        return

    def _setup_correct_topspin_importer(self):
        if len(self.expno_of_topspin_experiment) == 1:
            self.importer = io.Pseudo2DImporter(self)
        else:
            self.importer = io.ScreamImporter(self)

    def _read_in_data_from_topspin(self):
        self._setup_correct_topspin_importer()
        self.importer.import_topspin_data()



    def _read_in_data_from_csv(self):
        #TODO
        pass

    def _calculate_peak_intensities(self):
        self._add_peaks_to_all_exp()
        if "fit" in self.spectrum_fitting_type:
            self._perform_spectrum_fit()
        if "global" in self.spectrum_fitting_type:
            self._perform_global_spectrum_fit()

    def _buidup_fit_global(self):
        for type in self.buildup_type:
            buildup_type = ["biexponential"]
            if type == "biexponential":
                buildup_fitter = utils.BiexpFitter(self)
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
        print("h1")
        self.fitter.set_model()
        print("h2")
        self.fitter.fit()
        print("h3")
    def _get_intensities(self):
        pass

class Spectra():

    def __init__(self,file):
        self.file = file
        self.NS = ""
        self.tbup = ""
        self.x_axis = None
        self.y_axis = []
        self.peaks = []

    def add_peak(self,peak_dict):
        try:
            for peak in sorted(peak_dict.keys(), key=lambda x: int(x),
                       reverse=True):
                self.peaks.append(Peak(self,peak,peak_dict))
        except:
            print("ERROR: No peaks given. Try dataset.peak_dict = dict()")#TODO
            pass

class Peak():

    def __init__(self,spectrum,peak,peak_dict):
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

    def assign_values_from_dict(self):
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
            self.peak_label = f"Peak_at_{self.peak_center_rounded}_ppm_" \
                              f"{self.spectrum.tbup}_s"
            self.peak_label = self.peak_label.replace("-","m")

    def _set_fitting_group(self):
        try:
            self.fitting_group = int(self.peak_dict['fitting_group'])
        except:
            self.fitting_group = 999

    def _set_fitting_model(self):
        try:
            if self.peak_dict['fitting_model'] in ["voigt","gauss","lorentz"]:
                self.fitting_model = self.peak_dict['fitting_model']
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
        subspectrum = utils.generate_subspectrum(
            self.spectrum, self.peak_center_rounded, 4)

        if np.trapz(subspectrum) < 0:
            y_val = min(subspectrum)
        else:
            y_val = max(subspectrum)
        index = np.where(self.spectrum.y_axis == y_val)[0]
        x_val = self.spectrum.x_axis[index]
        self.hight = {'index':index[0],'x_val':x_val[0],'y_val':y_val}

class FileNameHandler():

    def __init__(self):
        self.path_to_topspin_experiment = None
        self.procno_of_topspin_experiment = None
        self.expno_of_topspin_experiment = None
        self.output_file_name = None

    def generate_output_csv_file_name(self):
        return self.output_file_name + ".csv"

    def generate_txt_fitting_report(self,peak_label,fitting_type):
        return f"{self.output_file_name}_{peak_label}_{fitting_type}.txt"

    def generate_spectrum_fit_pdf(self,fitting_type, tbup):
        output_folder = f"{self.output_file_name}_fit_per_spectrum/"
        os.makedirs(output_folder, exist_ok=True)
        return f"{output_folder}Delay_time_{tbup}" \
               f"_{fitting_type}.pdf"

    def generate_output_pdf_file_name(self):
        return self.output_file_name + ".pdf"