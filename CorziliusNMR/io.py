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
import copy


class TopspinImporter:
    """
    Class for importing NMR data from Bruker's TopSpin software.

    :param dataset: The dataset object to store imported spectra.
    :type dataset: CorziliusNMR.dataset.Dataset
    """

    def __init__(self, dataset):
        """
        Initialize the TopspinImporter.

        :param dataset: The dataset to store imported spectra.
        :type dataset: CorziliusNMR.dataset.Dataset
        """
        self._dataset = dataset
        self._top = top.Topspin()
        self._data_provider = self._top.getDataProvider()
        self._current_path_to_exp = None
        self._nmr_data = None

    def import_topspin_data(self):
        """
        Import NMR data from TopSpin.
        """
        pass

    def _set_values(self):
        """
        Set internal values including scans, buildup time, x and y data.
        """
        self._set_number_of_scans()
        self._set_buildup_time()
        self._set_x_data()
        self._set_y_data()
        self._normalize_y_values_to_number_of_scans()
        self._close()

    def _add_spectrum(self):
        """
        Add a new spectrum to the dataset.
        """
        self._dataset.spectra.append(CorziliusNMR.dataset.Spectra())

    def _get_physical_range(self):
        """
        Retrieve the physical range of the spectrum.

        :return: The physical range of the spectrum.
        :rtype: dict
        """
        return self._nmr_data.getSpecDataPoints()["physicalRanges"][0]

    def _get_num_of_datapoints(self):
        """
        Retrieve the number of data points in the spectrum.

        :return: Number of data points.
        :rtype: int
        """
        return len(self._nmr_data.getSpecDataPoints()["dataPoints"])

    def _calc_x_axis(self, physical_range, number_of_datapoints):
        """
        Calculate the x-axis values based on the physical range and number of data points.

        :param physical_range: The physical range of the spectrum.
        :type physical_range: dict
        :param number_of_datapoints: Number of data points in the spectrum.
        :type number_of_datapoints: int
        :return: Calculated x-axis values.
        :rtype: numpy.ndarray
        """
        return np.linspace(
            float(physical_range["start"]),
            float(physical_range["end"]),
            number_of_datapoints,
        )

    def _close(self):
        """
        Close the TopSpin API connection.
        """
        self._top.apiClient.pool.close()


class ScreamImporter(TopspinImporter):
    """
    Class for importing and processing Scream NMR data.
    """

    def import_topspin_data(self):
        """
        Import NMR data from TopSpin and process it.
        """
        files = self._generate_path_to_experiment()
        for file in files:
            self._add_spectrum()
            self._nmr_data = self._data_provider.getNMRData(file)
            self._set_values()

    def _set_number_of_scans(self):
        """
        Set the number of scans for the last spectrum in the dataset.
        """
        self._dataset.spectra[-1].number_of_scans = int(
            self._nmr_data.getPar("NS")
        )

    def _set_buildup_time(self):
        """
        Set the buildup time for the last spectrum in the dataset.
        """
        loop = float(self._nmr_data.getPar(self._dataset.props.loop20))
        delay = float(self._nmr_data.getPar(self._dataset.props.delay20))
        self._dataset.spectra[-1].tdel = loop * delay

    def _set_x_data(self):
        """
        Set the x-axis data for the last spectrum in the dataset.
        """
        physical_range = self._get_physical_range()
        number_of_datapoints = self._get_num_of_datapoints()
        self._dataset.spectra[-1].x_axis = self._calc_x_axis(
            physical_range, number_of_datapoints
        )

    def _set_y_data(self):
        """
        Set the y-axis data for the last spectrum in the dataset.
        """
        self._dataset.spectra[-1].y_axis = self._nmr_data.getSpecDataPoints()[
            "dataPoints"
        ]

    def _normalize_y_values_to_number_of_scans(self):
        """
        Normalize the y-axis values to the number of scans.
        """
        self._dataset.spectra[-1].y_axis = np.divide(
            self._dataset.spectra[-1].y_axis,
            self._dataset.spectra[-1].number_of_scans,
        )

    def _generate_path_to_experiment(self):
        """
        Generate file paths for all experiment numbers.

        :return: List of file paths to experiment data.
        :rtype: list
        """
        base_path = self._dataset.props.path_to_experiment
        procno = self._dataset.props.procno
        path_list = [
            os.path.join(base_path, str(expno), "pdata", str(procno))
            for expno in self._dataset.props.expno
        ]
        return path_list


class Pseudo2DImporter(TopspinImporter):
    """
    Class for importing and processing pseudo-2D NMR data.
    """

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
            self._plot_global_each_individual()
        for buildup_type in self.dataset.props.buildup_types:
            self._plot_buildup(buildup_type)
        self._print_report()

    def _plot_topspin_data(self):
        colormap = plt.cm.viridis
        colors = [
            colormap(i / len(self.dataset.spectra))
            for i in range(len(self.dataset.spectra))
        ]
        for idx, spectrum in enumerate(self.dataset.spectra):
            plt.plot(
                spectrum.x_axis,
                spectrum.y_axis,
                label=f"t_del = {spectrum.tdel} s",
                color=colors[idx],
            )
        plt.gca().invert_xaxis()
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
                simspec = functions.calc_peak(x_axis, simspec, val)

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
        axs[1].set_ylabel("$I_{resid}$ / a.u.")
        axs[0].set_xlim(max(x_axis), min(x_axis))
        axs[1].set_xlim(max(x_axis), min(x_axis))
        axs[1].legend()
        plt.tight_layout()
        # plt.show()
        plt.close()

    def _print_lmfit_prefit_report(self):
        pass

    def _plot_global(self):
        valdict = functions.generate_spectra_param_dict(
            self.dataset.lmfit_result_handler.global_fit.params
        )
        fig, ax = plt.subplots(1, 1, sharex=True)
        first = True
        for keys in valdict:
            simspec = np.zeros_like(
                self.dataset.spectra[keys].y_axis, dtype=float
            )
            for val in valdict[keys]:
                simspec = functions.calc_peak(
                    self.dataset.spectra[keys].x_axis, simspec, val
                )
            label_exp = "Experiment" if first else None
            label_sim = "Simulation" if first else None
            ax.plot(
                self.dataset.spectra[keys].x_axis,
                self.dataset.spectra[keys].y_axis,
                color="black",
                label=label_exp,
            )
            ax.plot(
                self.dataset.spectra[keys].x_axis,
                simspec,
                "r--",
                label=label_sim,
            )
            first = False
        ax.legend()
        ax.invert_xaxis()
        ax.set_ylabel("$I$ / a.u.")
        plt.tight_layout()
        # plt.show()
        plt.close()

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

    def _plot_global_each_individual(self):
        valdict = functions.generate_spectra_param_dict(
            self.dataset.lmfit_result_handler.global_fit.params
        )
        for keys in valdict:
            simspec = [
                0 for _ in range(len(self.dataset.spectra[keys].y_axis))
            ]
            fig, axs = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
            )
            residual = residual = copy.deepcopy(
                self.dataset.spectra[keys].y_axis
            )
            for val in valdict[keys]:
                simspec = functions.calc_peak(
                    self.dataset.spectra[keys].x_axis, simspec, val
                )
                axs[0].plot(
                    self.dataset.spectra[keys].x_axis,
                    self.dataset.spectra[keys].y_axis,
                    color="black",
                    label="Experiment",
                )

            residual -= simspec
            axs[0].plot(
                self.dataset.spectra[keys].x_axis,
                simspec,
                "r--",
                label="Simulation",
            )
            axs[1].plot(
                self.dataset.spectra[keys].x_axis,
                residual,
                color="grey",
                label="Residual",
            )

            axs[0].set_ylabel("$I$ / a.u.")
            axs[1].set_xlabel("$\delta$ / ppm")
            axs[1].set_ylabel("$I_{resid}$ / a.u.")
            axs[0].set_xlim(
                max(self.dataset.spectra[keys].x_axis),
                min(self.dataset.spectra[keys].x_axis),
            )
            axs[1].set_xlim(
                max(self.dataset.spectra[keys].x_axis),
                min(self.dataset.spectra[keys].x_axis),
            )
            axs[0].legend()
            axs[1].legend()
            plt.tight_layout()
            # plt.show()
            plt.close()

    def _print_report(self, filename="report.txt"):
        with open(filename, "w") as f:
            f.write(str(self.dataset.props) + "\n")
            f.write(str(self.dataset) + "\n")
            f.write("[[Peaks]]\n")
            for peak_nr, peak in enumerate(self.dataset.peak_list):
                f.write(f"[Peak {peak_nr + 1}]\n")
                f.write(str(peak) + "\n")

            if self.dataset.props.prefit:
                f.write("[[Prefit]]\n")
                valdict = functions.generate_spectra_param_dict(
                    self.dataset.lmfit_result_handler.prefit.params
                )
                f.write(
                    "Label\t\t\tCenter\t\t\t\tAmplitude\t\t\tSigma\t\t\t\tGamma\n"
                )
                for key_nr, keys in enumerate(valdict):
                    for val in valdict[keys]:
                        if len(val) == 5:
                            f.write(
                                f"{self.dataset.peak_list[key_nr].peak_label}\t{val[1]}\t{val[0]}\t{val[2]}\t{val[3]}\n"
                            )
                        elif len(val) == 3:
                            f.write(
                                f"{self.dataset.peak_list[key_nr].peak_label}\t{val[1]}\t{val[0]}\t{val[2]}\t---\n"
                            )
                        elif len(val) == 4:
                            f.write(
                                f"{self.dataset.peak_list[key_nr].peak_label}\t{val[1]}\t{val[0]}\t---\t{val[2]}\n"
                            )
            else:
                f.write("[[Prefit]]\nNo prefit performed.\n")

            f.write("[[Global fit results]]\n")
            valdict = functions.generate_spectra_param_dict(
                self.dataset.lmfit_result_handler.global_fit.params
            )
            header = [
                "Label",
                "Time",
                "Center",
                "Amplitude",
                "Sigma",
                "Gamma",
                "FWHM Lorentz",
                "FWHM Gauss",
                "FWHM Voigt",
            ]
            column_widths = [25, 6, 25, 25, 25, 25, 25, 25, 25]
            f.write(
                "".join(f"{h:<{w}}" for h, w in zip(header, column_widths))
                + "\n"
            )
            for val_nr, (key, values) in enumerate(valdict.items()):
                for val_index, val in enumerate(values):
                    if len(val) == 5:
                        row = [
                            self.dataset.peak_list[val_index].peak_label,
                            self.dataset.spectra[val_nr].tdel,
                            val[1],
                            val[0],
                            val[2],
                            val[3],
                            functions.fwhm_lorentzian(val[3]),
                            functions.fwhm_gaussian(val[2]),
                            functions.fwhm_voigt(val[2], val[3]),
                        ]
                        f.write(
                            "".join(
                                f"{str(item):<{w}}"
                                for item, w in zip(row, column_widths)
                            )
                            + "\n"
                        )

            f.write("[[Buildup fit results]]\n")
            for type in self.dataset.props.buildup_types:
                f.write(f"[{type}]\n")
                header = [
                    "Label",
                    "A1 / a.u.",
                    "t1 / s",
                    "A2 / a.u.",
                    "t2 / s",
                    "t_off / s",
                ]
                column_widths = [20] * len(header)
                f.write(
                    "".join(h.ljust(w) for h, w in zip(header, column_widths))
                    + "\n"
                )
                format_mappings = {
                    "exponential": ["A1", "t1", "---", "---", "---"],
                    "exponential_with_offset": [
                        "A1",
                        "t1",
                        "---",
                        "---",
                        "x1",
                    ],
                    "biexponential": ["A1", "t1", "A2", "t2", "---"],
                    "biexponential_with_offset": [
                        "A1",
                        "t1",
                        "A2",
                        "t2",
                        "x1",
                    ],
                }
                type_format = format_mappings.get(type, [])
                for result_nr, result in enumerate(
                    self.dataset.lmfit_result_handler.buildup_fit[type]
                ):
                    row_data = [self.dataset.peak_list[result_nr].peak_label]
                    for param in type_format:
                        value = (
                            str(result.params[param].value)
                            if param != "---"
                            else "---"
                        )
                        row_data.append(value)
                    f.write(
                        "".join(cell.ljust(20) for cell in row_data) + "\n"
                    )
            f.write("End\n")


class LmfitResultHandler:
    def __init__(self):
        self.prefit = None
        self.single_fit = None
        self.global_fit = None
        self.buildup_fit = {}
