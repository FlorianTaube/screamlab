"""
io module of the CorziliusNMR package.
"""

import CorziliusNMR.dataset
from CorziliusNMR import utils, functions
import numpy as np
import bruker.api.topspin as top
import os
import matplotlib.pyplot as plt
import lmfit
import matplotlib.cm as cm


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
        self._close()

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

    def _close(self):
        self._top.apiClient.pool.close()


class ScreamImporter(TopspinImporter):

    def import_topspin_data(self):
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


class Exporter:

    def __init__(self, dataset):
        self.dataset = dataset

    def print(self):
        self._plot_topspin_data()
        if self.dataset.props.prefit:
            self._plot_prefit()
            self._print_lmfit_prefit_report()
        if "global" in self.dataset.props.spectrum_fit_type:
            self._plot_global()
            self._plot_global_all_in_one()
        for buildup_type in self.dataset.props.buildup_types:
            self._plot_buildup(buildup_type)

    def _plot_topspin_data(self):
        for spectrum in self.dataset.spectra:
            plt.plot(
                spectrum.x_axis,
                spectrum.y_axis,
                label=f"t_del = {spectrum.tdel} s",
            )
        plt.legend()
        # plt.show()
        plt.close()

    def _plot_prefit(self):
        x_axis = self.dataset.spectra[
            self.dataset.props.spectrum_for_prefit
        ].x_axis
        y_axis = self.dataset.spectra[
            self.dataset.props.spectrum_for_prefit
        ].y_axis

        valdict = functions.generate_spectra_param_dict(
            self.dataset.lmfit_result_handler.prefit.params
        )
        simspec = [0 for _ in range(len(y_axis))]
        for keys in valdict:
            for val in valdict[keys]:
                simspec += functions.calc_peak(x_axis, simspec, val)

        fig, axs = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        axs[0].plot(x_axis, y_axis, color="black", label="Experiment")

        axs[0].plot(x_axis, simspec, "r--", label="Simulation")
        axs[0].legend()
        axs[0].set_ylabel("$I$ / a.u.")
        residual = y_axis - simspec
        axs[1].plot(x_axis, residual, color="grey", label="Residual")
        axs[1].set_xlabel("$\delta$ / ppm")
        axs[1].set_ylabel("Residual")
        axs[1].legend()

        plt.tight_layout()
        plt.show()
        plt.close()

    def _print_lmfit_prefit_report(self):
        pass

    def _plot_global(self):
        pass

    def _plot_buildup(self, buildup_type):
        colors = plt.colormaps.get_cmap("viridis")  # Hol dir die Colormap
        norm = plt.Normalize(vmin=0, vmax=len(self.dataset.peak_list))
        for peak_nr, peak in enumerate(self.dataset.peak_list):
            peak_result = self.dataset.lmfit_result_handler.buildup_fit[
                buildup_type
            ][peak_nr]
            color = colors(norm(peak_nr))
            plt.plot(
                peak.buildup_vals.tdel,
                peak.buildup_vals.intensity,
                "o",
                color=color,
                label=f"{peak.peak_label}",
            )
            sim_tdel = np.linspace(0, peak.buildup_vals.tdel[-1], 1024)
            val_list = [param.value for param in peak_result.params.values()]

            func_map = {
                "exponential": functions.calc_exponential,
                "biexponential": functions.calc_biexponential,
                "exponential_with_offset": functions.calc_exponential_with_offset,
                "biexponential_with_offset": functions.calc_biexponential_with_offset,
            }
            plt.plot(
                sim_tdel,
                func_map[buildup_type](sim_tdel, val_list),
                "-",
                color=color,
            )
        plt.legend()
        # plt.show()
        plt.close()

    def _plot_global_all_in_one(self):
        pass


class LmfitResultHandler:
    def __init__(self):
        self.prefit = None
        self.single_fit = None
        self.global_fit = None
        self.buildup_fit = {}
