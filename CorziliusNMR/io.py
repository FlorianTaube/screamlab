"""
io module of the CorziliusNMR package.
"""

import CorziliusNMR.dataset
from CorziliusNMR import utils
import numpy as np
from bruker.data.nmr import *
import bruker.api.topspin as top
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import os

from tabulate import tabulate


class TopspinImporter:

    def __init__(self, dataset):
        self._dataset = dataset
        self._top = top.Topspin()
        self._data_provider = self._top.getDataProvider()
        self._current_path_to_exp = None
        self._nmr_data = None

    def import_topspin_data(self):
        pass

    def _set_values(self):  # TODO Test
        self._set_number_of_scans()
        self._set_buildup_time()
        self._set_x_data()
        self._set_y_axis()
        self._normalize_y_values_to_number_of_scans()

    def _set_number_of_scans(self):
        pass

    def _set_buildup_time(self):
        pass

    def _set_x_data(self):
        pass

    def _set_y_data(self):
        pass

    def _normalize_y_values_to_number_of_scans(self):
        pass

    def _generate_path_to_experiment(self):
        pass

    def _add_spectrum(self):
        self._dataset.spectra.append(CorziliusNMR.dataset.Spectra())


class ScreamImporter(TopspinImporter):

    def __init__(self, dataset):
        super().__init__(dataset)

    def import_topspin_data(self):  # TODO Test
        files = self._generate_path_to_experiment()
        for file in files:
            self._add_spectrum()
            self._nmr_data = self._data_provider.getNMRData(file)
            self._set_values()

    def _add_spectrum(self):
        super()._add_spectrum()

    def _set_values(self):
        super()._set_values()

    def _set_number_of_scans(self):
        self._dataset.spectra[-1].number_of_scans = int(
            self._nmr_data.getPar("NS")
        )

    def _set_buildup_time(self):
        self._dataset.spectra[-1].tbup = (
            int(self._nmr_data.getPar("L 20")) / 4
        )

    def _set_x_data(self):
        _physicalRange = self._nmr_data.getSpecDataPoints()["physicalRanges"][
            0
        ]
        _number_of_datapoints = len(
            self._nmr_data.getSpecDataPoints()["dataPoints"]
        )
        axis = np.linspace(
            float(_physicalRange["start"]),
            float(_physicalRange["end"]),
            _number_of_datapoints,
        )
        self._dataset.spectra[-1].x_axis = utils.generate_subspectrum_2(
            axis,
            axis,
            max(list(map(int, self._dataset.peak_dict.keys()))),
            min(list(map(int, self._dataset.peak_dict.keys()))),
            50,
        )
        if self._len == 0:
            self._len = len(self._dataset.spectra[-1].x_axis)
        else:
            if self._len > len(self._dataset.spectra[-1].x_axis):
                diff = self._len - len(self._dataset.spectra[-1].x_axis)
                count = 0
                while count < diff:
                    self._dataset.spectra[-1].x_axis = np.append(
                        self._dataset.spectra[-1].x_axis, 0
                    )
                    count += 1

    def _set_y_axis(self):
        axis = self._nmr_data.getSpecDataPoints()["dataPoints"]
        _physicalRange = self._nmr_data.getSpecDataPoints()["physicalRanges"][
            0
        ]
        _number_of_datapoints = len(
            self._nmr_data.getSpecDataPoints()["dataPoints"]
        )
        x_axis = np.linspace(
            float(_physicalRange["start"]),
            float(_physicalRange["end"]),
            _number_of_datapoints,
        )
        self._dataset.spectra[-1].y_axis = utils.generate_subspectrum_2(
            x_axis,
            axis,
            max(list(map(int, self._dataset.peak_dict.keys()))),
            min(list(map(int, self._dataset.peak_dict.keys()))),
            50,
        )
        if self._len == 0:
            self._len = len(self._dataset.spectra[-1].y_axis)
        else:
            if self._len > len(self._dataset.spectra[-1].y_axis):
                diff = self._len - len(self._dataset.spectra[-1].y_axis)
                count = 0
                while count < diff:
                    self._dataset.spectra[-1].y_axis = np.append(
                        self._dataset.spectra[-1].y_axis, 0
                    )
                    count += 1

    def _normalize_y_values_to_number_of_scans(self):
        self._dataset.spectra[-1].y_axis = np.divide(
            self._dataset.spectra[-1].y_axis, self._dataset.spectra[-1].NS
        )

    def _generate_path_to_experiment(self):
        base_path = self._dataset.props.path_to_experiment
        procno = self._dataset.props.procno
        return [
            f"{base_path}/{expno}/pdata/{procno}".replace("\\", "/")
            for expno in self._dataset.props.expno
        ]


class Pseudo2DImporter(TopspinImporter):

    def __init__(self, dataset):
        super().__init__(dataset)

    def import_topspin_data(self):
        return

    def read_in_topspin_data():
        pass
