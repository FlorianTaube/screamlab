"""
io module of the CorziliusNMR package.
"""

import CorziliusNMR.dataset
import numpy as np
import bruker.api.topspin as top
import os


class TopspinImporter:

    def __init__(self, dataset):
        self._dataset = dataset
        self._top = top.Topspin()
        self._data_provider = self._top.getDataProvider()
        self._current_path_to_exp = None
        self._nmr_data = None

    def import_topspin_data(self):
        pass

    def _set_values(self):
        self._set_number_of_scans()
        self._set_buildup_time()
        self._set_x_data()
        self._set_y_data()
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

    def _get_physical_range(self):
        return self._nmr_data.getSpecDataPoints()["physicalRanges"][0]

    def _get_num_of_datapoints(self):
        return len(self._nmr_data.getSpecDataPoints()["dataPoints"])

    def _calc_x_axis(self, physical_range, number_of_datapoints):
        return np.linspace(
            float(physical_range["start"]),
            float(physical_range["end"]),
            number_of_datapoints,
        )


class ScreamImporter(TopspinImporter):

    def import_topspin_data(self):  # TODO Test
        files = self._generate_path_to_experiment()
        for file in files:
            self._add_spectrum()
            self._nmr_data = self._data_provider.getNMRData(file)
            self._set_values()

    def _set_number_of_scans(self):
        self._dataset.spectra[-1].number_of_scans = int(
            self._nmr_data.getPar("NS")
        )

    def _set_buildup_time(self):
        loop = float(self._nmr_data.getPar(self._dataset.props.loop20))
        delay = float(self._nmr_data.getPar(self._dataset.props.delay20))
        self._dataset.spectra[-1].tdel = loop * delay

    def _set_x_data(self):
        physical_range = self._get_physical_range()
        number_of_datapoints = self._get_num_of_datapoints()
        self._dataset.spectra[-1].x_axis = self._calc_x_axis(
            physical_range, number_of_datapoints
        )

    def _set_y_data(self):
        self._dataset.spectra[-1].y_axis = self._nmr_data.getSpecDataPoints()[
            "dataPoints"
        ]

    def _normalize_y_values_to_number_of_scans(self):
        self._dataset.spectra[-1].y_axis = np.divide(
            self._dataset.spectra[-1].y_axis,
            self._dataset.spectra[-1].number_of_scans,
        )

    def _generate_path_to_experiment(self):
        base_path = self._dataset.props.path_to_experiment
        procno = self._dataset.props.procno
        path_list = [
            os.path.join(base_path, str(expno), "pdata", str(procno))
            for expno in self._dataset.props.expno
        ]
        return path_list


class Pseudo2DImporter(TopspinImporter):
    pass


class LmfitResultHandler:
    def __init__(self):
        self.prefit = None
        self.single_fit = None
        self.global_fit = None
        self.buidlup_fit = {}
