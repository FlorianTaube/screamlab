from CorziliusNMR import io,plotter
import numpy as np
import sys


class Dataset:

    def __init__(self):
        self._fileNames = io._FileNameHandler()
        self._topspin_exporter = None
        self.experiments = []

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
        self._read_in_data_from_topspin(self)
        self._calculate_peak_intensities(self)



    def start_buildup_fit_from_spectra(self):
        self._read_in_data_from_csv(self)
        self._calculate_peak_intensities(self)
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

    def _calculate_peak_intensities(self):
        self._set_peak_pos(self)
        self._set_peak_sign(self)
        self._set_peak_label(self)



