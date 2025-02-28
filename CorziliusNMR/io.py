"""
io module of the CorziliusNMR package.
"""

from CorziliusNMR import dataset, functions
import numpy as np
import bruker.api.topspin as top
import os
import matplotlib.pyplot as plt
import copy
import csv
import math
import sys


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
        self._dataset.spectra.append(dataset.Spectra())

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


class ScreamImporter(TopspinImporter):
    """
    Class for importing and processing Scream NMR data.
    """

    def _set_number_of_scans(self):
        """
        Set the number of scans for the last spectrum in the dataset.
        """
        self._dataset.spectra[-1].number_of_scans = int(
            self._nmr_data.getPar("NS")
        )

    def import_topspin_data(self):
        """
        Import NMR data from TopSpin and process it.
        """
        files = self._generate_path_to_experiment()
        for file in files:
            self._add_spectrum()
            self._nmr_data = self._data_provider.getNMRData(file)
            self._set_values()

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


class Pseudo2DImporter(TopspinImporter):
    pass

    def import_topspin_data(self):
        """
        Import NMR data from TopSpin and process it.
        """
        files = self._generate_path_to_experiment()
        vdlist = os.path.join(
            self._dataset.props.path_to_experiment,
            str(self._dataset.props.expno[0]),
            "vdlist",
        )
        vdfile = open(vdlist, "r")
        for tdel in vdfile:
            self._add_spectrum()
            self._dataset.spectra[-1].tdel = float(tdel)
        vdfile.close()

        self._nmr_data = self._data_provider.getNMRData(files[0])
        for spectrum_nr in range(
            0,
            self._nmr_data.getSpecDataPoints()["indexRanges"][1][
                "numberOfPoints"
            ],
        ):
            physical_range = self._nmr_data.getSpecDataPoints()[
                "physicalRanges"
            ][0]
            number_of_datapoints = self._nmr_data.getSpecDataPoints()[
                "indexRanges"
            ][0]["numberOfPoints"]
            self._dataset.spectra[spectrum_nr].x_axis = self._calc_x_axis(
                physical_range, number_of_datapoints
            )
            start = 0 + spectrum_nr * number_of_datapoints
            stop = number_of_datapoints + spectrum_nr * number_of_datapoints
            self._dataset.spectra[spectrum_nr].number_of_scans = (
                self._nmr_data.getPar("NS")
            )
            self._dataset.spectra[
                spectrum_nr
            ].y_axis = self._nmr_data.getSpecDataPoints()["dataPoints"][
                start:stop
            ]
        self._close()


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
        self._write_global_fit_results_to_semicolon_separated_something()
        self._write_buildup_fit_to_semicolon_separated_something()
        self._csv_output()

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
        plt.xlabel("$t_{del}$ / s")
        plt.ylabel("$I$ / a.u.")
        plt.legend()
        plt.savefig(
            f"{self.dataset.props.output_folder}/Exported_data.pdf",
            dpi=500,
            bbox_inches="tight",
        )
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
        for _, values in valdict.items():
            for val in values:
                simspec = functions.calc_peak(x_axis, simspec, val)

        _, axs = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        axs[0].plot(x_axis, y_axis, color="black", label="Experiment")

        axs[0].plot(x_axis, simspec, "r--", label="Simulation")
        axs[0].legend()
        axs[0].set_ylabel("$I$ / a.u.")
        residual = y_axis - simspec
        axs[1].plot(x_axis, residual, color="grey", label="Residual")
        axs[1].set_xlabel("$\\delta$ / ppm")
        axs[1].set_ylabel("$I_{resid}$ / a.u.")
        axs[0].set_xlim(max(x_axis), min(x_axis))
        axs[1].set_xlim(max(x_axis), min(x_axis))
        axs[1].legend()
        plt.xlabel("$t_{del}$ / s")
        plt.ylabel("$I$ / a.u.")
        plt.tight_layout()
        plt.savefig(
            f"{self.dataset.props.output_folder}/Prefit_plot.pdf",
            dpi=500,
            bbox_inches="tight",
        )
        plt.close()

    def _print_lmfit_prefit_report(self):
        pass

    def _plot_global(self):
        valdict = functions.generate_spectra_param_dict(
            self.dataset.lmfit_result_handler.global_fit.params
        )
        _, ax = plt.subplots(1, 1, sharex=True)
        first = True
        for key, values in valdict.items():
            simspec = np.zeros_like(
                self.dataset.spectra[key].y_axis, dtype=float
            )
            for val in values:
                simspec = functions.calc_peak(
                    self.dataset.spectra[key].x_axis, simspec, val
                )
            label_exp = "Experiment" if first else None
            label_sim = "Simulation" if first else None
            ax.plot(
                self.dataset.spectra[key].x_axis,
                self.dataset.spectra[key].y_axis,
                color="black",
                label=label_exp,
            )
            ax.plot(
                self.dataset.spectra[key].x_axis,
                simspec,
                "r--",
                label=label_sim,
            )
            first = False
        plt.xlabel("$t_{del}$ / s")
        plt.ylabel("$I$ / a.u.")
        ax.legend()
        ax.invert_xaxis()
        ax.set_ylabel("$I$ / a.u.")
        plt.tight_layout()
        plt.savefig(
            f"{self.dataset.props.output_folder}/Global_fit_summary.pdf",
            dpi=500,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_buildup(self, buildup_type):
        colors = plt.get_cmap("viridis")
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
        plt.xlabel("$t_{del}$ / s")
        plt.ylabel("$I$ / a.u.")
        plt.legend()
        plt.savefig(
            f"{self.dataset.props.output_folder}/Buildup_fit_{buildup_type}.pdf",
            dpi=500,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_global_each_individual(self):
        output_dir = os.path.join(
            self.dataset.props.output_folder, "fit_per_spectrum"
        )
        os.makedirs(output_dir, exist_ok=True)

        param_dict = functions.generate_spectra_param_dict(
            self.dataset.lmfit_result_handler.global_fit.params
        )

        for key, param_list in param_dict.items():
            spectrum = self.dataset.spectra[key]
            x_axis, y_axis = spectrum.x_axis, spectrum.y_axis

            simspec = [0] * len(y_axis)
            residual = copy.deepcopy(y_axis)

            fig, axs = plt.subplots(
                2,
                1,
                sharex=True,
                sharey=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

            for params in param_list:
                simspec = functions.calc_peak(x_axis, simspec, params)

            axs[0].plot(x_axis, y_axis, color="black", label="Experiment")
            axs[0].plot(x_axis, simspec, "r--", label="Simulation", alpha=0.8)

            residual -= simspec
            axs[1].plot(x_axis, residual, color="grey", label="Residual")

            axs[0].set_ylabel("$I$ / a.u.")
            axs[1].set_xlabel("$\\delta$ / ppm")
            axs[1].set_ylabel("$I_{resid}$ / a.u.")

            for ax in axs:
                ax.set_xlim(max(x_axis), min(x_axis))
                ax.legend()

            plt.tight_layout()
            plot_filename = os.path.join(
                output_dir, f"Spectrum_at_{spectrum.tdel}_s.pdf"
            )
            plt.savefig(plot_filename, dpi=500, bbox_inches="tight")
            plt.close(fig)

    def _print_report(self):
        with open(
            f"{self.dataset.props.output_folder}/Analysis_result.txt",
            "w",
            encoding="utf-8",
        ) as f:
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
                    for val_nr, val in enumerate(valdict[keys]):
                        if len(val) == 5:
                            f.write(
                                f"{self.dataset.peak_list[val_nr].peak_label}\t{val[1]}\t{val[0]}\t{val[2]}\t{val[3]}\n"
                            )
                        elif len(val) == 3:
                            f.write(
                                f"{self.dataset.peak_list[val_nr].peak_label}\t{val[1]}\t{val[0]}\t{val[2]}\t---\n"
                            )
                        elif len(val) == 4:
                            f.write(
                                f"{self.dataset.peak_list[val_nr].peak_label}\t{val[1]}\t{val[0]}\t---\t{val[2]}\n"
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
                "Integral",
            ]
            column_widths = [25, 6, 10, 15, 10, 10, 15, 15, 15, 10]
            f.write(
                "".join(f"{h:<{w}}" for h, w in zip(header, column_widths))
                + "\n"
            )

            for delay_time in range(0, len(valdict[0])):
                for val_nr, (_, values) in enumerate(valdict.items()):
                    if len(values[delay_time]) == 5:
                        row = [
                            self.dataset.peak_list[delay_time].peak_label,
                            self.dataset.spectra[val_nr].tdel,
                            round(values[delay_time][1], 3),
                            round(values[delay_time][0], 3),
                            round(values[delay_time][2], 3),
                            round(values[delay_time][3], 3),
                            round(
                                functions.fwhm_lorentzian(
                                    values[delay_time][3]
                                ),
                                3,
                            ),
                            round(
                                functions.fwhm_gaussian(
                                    values[delay_time][2]
                                ),
                                3,
                            ),
                            round(
                                functions.fwhm_voigt(
                                    values[delay_time][2],
                                    values[delay_time][3],
                                ),
                                3,
                            ),
                            round(
                                self.dataset.peak_list[
                                    delay_time
                                ].buildup_vals.intensity[val_nr],
                                3,
                            ),
                        ]
                        f.write(
                            "".join(
                                f"{str(item):<{w}}"
                                for item, w in zip(row, column_widths)
                            )
                            + "\n"
                        )

            f.write("[[Buildup fit results]]\n")
            for buildup_type in self.dataset.props.buildup_types:
                f.write(f"[{buildup_type}]\n")
                header = [
                    "Label",
                    "A1 / a.u.",
                    "t1 / s",
                    "A2 / a.u.",
                    "t2 / s",
                    "t_off / s",
                    "R1 / 1/s",
                    "R2 / 1/s",
                    "Sensitivity1 (A1/sqrt(t1)",
                    "Sensitivity2 (A2/sqrt(t2)",
                ]
                column_widths = [20, 15, 10, 15, 10, 15, 15, 15, 35, 35]
                f.write(
                    "".join(h.ljust(w) for h, w in zip(header, column_widths))
                    + "\n"
                )
                format_mappings = {
                    "exponential": [
                        "A1",
                        "t1",
                        "---",
                        "---",
                        "---",
                        "R1",
                        "---",
                        "S1",
                        "---",
                    ],
                    "exponential_with_offset": [
                        "A1",
                        "t1",
                        "---",
                        "---",
                        "x1",
                        "R1",
                        "---",
                        "S1",
                        "---",
                    ],
                    "biexponential": [
                        "A1",
                        "t1",
                        "A2",
                        "t2",
                        "---",
                        "R1",
                        "R2",
                        "S1",
                        "S2",
                    ],
                    "biexponential_with_offset": [
                        "A1",
                        "t1",
                        "A2",
                        "t2",
                        "x1",
                        "R1",
                        "R2",
                        "S1",
                        "S2",
                    ],
                }
                type_format = format_mappings.get(buildup_type, [])
                for result_nr, result in enumerate(
                    self.dataset.lmfit_result_handler.buildup_fit[
                        buildup_type
                    ]
                ):
                    row_data = [self.dataset.peak_list[result_nr].peak_label]

                    param_calculations = {
                        "R1": lambda: str(round(1 / float(row_data[2]), 5)),
                        "R2": lambda: str(round(1 / float(row_data[4]), 5)),
                        "S1": lambda: str(
                            round(
                                float(row_data[1])
                                / math.sqrt(float(row_data[2])),
                                3,
                            )
                        ),
                        "S2": lambda: str(
                            round(
                                float(row_data[3])
                                / math.sqrt(float(row_data[4])),
                                3,
                            )
                        ),
                    }
                    for param in type_format:
                        if param == "---":
                            value = "---"
                        elif param in param_calculations:
                            value = param_calculations[param]()
                        else:
                            value = str(round(result.params[param].value, 3))
                        row_data.append(value)
                    f.write(
                        "".join(
                            h.ljust(w)
                            for h, w in zip(row_data, column_widths)
                        )
                        + "\n"
                    )
            f.write("End\n")

    def _write_buildup_fit_to_semicolon_separated_something(self):
        for buildup_type in self.dataset.props.buildup_types:
            with open(
                f"{self.dataset.props.output_folder}/Buildup_fit_result_{buildup_type}.txt",
                "w",
                encoding="utf-8",
            ) as f:
                header = [
                    "Label",
                    "A1 / a.u.",
                    "t1 / s",
                    "A2 / a.u.",
                    "t2 / s",
                    "t_off / s",
                    "R1 / 1/s",
                    "R2 / 1/s",
                    "Sensitivity1 (A1/sqrt(t1))",
                    "Sensitivity2 (A2/sqrt(t2))",
                ]
                f.write(";".join(header) + "\n")

                format_mappings = {
                    "exponential": [
                        "A1",
                        "t1",
                        "---",
                        "---",
                        "---",
                        "R1",
                        "---",
                        "S1",
                        "---",
                    ],
                    "exponential_with_offset": [
                        "A1",
                        "t1",
                        "---",
                        "---",
                        "x1",
                        "R1",
                        "---",
                        "S1",
                        "---",
                    ],
                    "biexponential": [
                        "A1",
                        "t1",
                        "A2",
                        "t2",
                        "---",
                        "R1",
                        "R2",
                        "S1",
                        "S2",
                    ],
                    "biexponential_with_offset": [
                        "A1",
                        "t1",
                        "A2",
                        "t2",
                        "x1",
                        "R1",
                        "R2",
                        "S1",
                        "S2",
                    ],
                }

                type_format = format_mappings.get(buildup_type, [])

                for result_nr, result in enumerate(
                    self.dataset.lmfit_result_handler.buildup_fit[
                        buildup_type
                    ]
                ):
                    row_data = [self.dataset.peak_list[result_nr].peak_label]

                    param_calculations = {
                        "R1": lambda: str(round(1 / float(row_data[2]), 5)),
                        "R2": lambda: str(round(1 / float(row_data[4]), 5)),
                        "S1": lambda: str(
                            round(
                                float(row_data[1])
                                / math.sqrt(float(row_data[2])),
                                3,
                            )
                        ),
                        "S2": lambda: str(
                            round(
                                float(row_data[3])
                                / math.sqrt(float(row_data[4])),
                                3,
                            )
                        ),
                    }

                    for param in type_format:
                        if param == "---":
                            value = "---"
                        elif param in param_calculations:
                            value = param_calculations[param]()
                        else:
                            value = str(round(result.params[param].value, 3))
                        row_data.append(value)

                    f.write(";".join(row_data) + "\n")

    def _write_global_fit_results_to_semicolon_separated_something(self):
        with open(
            f"{self.dataset.props.output_folder}/Global_fit_result.txt",
            "w",
            encoding="utf-8",
        ) as f:
            valdict = functions.generate_spectra_param_dict(
                self.dataset.lmfit_result_handler.global_fit.params
            )
            header = [
                "Label",
                "Delay Time / s",
                "Center / ppm",
                "Amplitude / a.u.",
                "Sigma / ppm",
                "Gamma / ppm",
                "FWHM Lorentz / ppm",
                "FWHM Gauss / ppm",
                "FWHM Voigt / ppm",
                "Integral / a.u.",
            ]
            f.write(";".join(str(item) for item in header) + "\n")
            for delay_time in range(0, len(valdict[0])):
                for val_nr, (_, values) in enumerate(valdict.items()):
                    if len(values[delay_time]) == 5:
                        row = [
                            (
                                self.dataset.peak_list[delay_time].peak_label
                                if val_nr == 0
                                else ""
                            ),
                            self.dataset.spectra[val_nr].tdel,
                            round(values[delay_time][1], 3),
                            round(values[delay_time][0], 3),
                            round(values[delay_time][2], 3),
                            round(values[delay_time][3], 3),
                            round(
                                functions.fwhm_lorentzian(
                                    values[delay_time][3]
                                ),
                                3,
                            ),
                            round(
                                functions.fwhm_gaussian(
                                    values[delay_time][2]
                                ),
                                3,
                            ),
                            round(
                                functions.fwhm_voigt(
                                    values[delay_time][2],
                                    values[delay_time][3],
                                ),
                                3,
                            ),
                            round(
                                self.dataset.peak_list[
                                    delay_time
                                ].buildup_vals.intensity[val_nr],
                                3,
                            ),
                        ]
                        f.write(";".join(str(item) for item in row) + "\n")

    def _csv_output(self):
        spectral_data = []
        with open(
            f"{self.dataset.props.output_folder}\Spectral_data_csv.csv",
            "w",
            newline="",
        ) as file:
            for spectrum in self.dataset.spectra:
                spectral_data.append(spectrum.x_axis)
                spectral_data.append(spectrum.y_axis)
            writer = csv.writer(file, delimiter=";")
            for row in zip(*spectral_data):
                writer.writerow(row)


class LmfitResultHandler:
    """
    A class to handle the results of fitting operations.

    This class stores and manages the results from different types of fits:
    prefit, single fit, global fit, and buildup fit. It provides a container
    for the various fit results to facilitate later analysis and processing.

    Attributes:
        prefit (object or None): Stores the prefit result, which may be an object or None.
        single_fit (object or None): Stores the result of a single fit operation, or None if not available.
        global_fit (object or None): Stores the result of a global fit operation, or None if not available.
        buildup_fit (dict): A dictionary that stores results from buildup fits, keyed by fit identifiers.
    """

    def __init__(self):
        """
        Initializes the LmfitResultHandler with default values.

        The prefit, single_fit, and global_fit attributes are set to None,
        indicating that no fit results have been stored yet. The buildup_fit
        attribute is initialized as an empty dictionary to store multiple buildup fit results.

        Attributes:
            prefit (None): Default value for the prefit result.
            single_fit (None): Default value for the single fit result.
            global_fit (None): Default value for the global fit result.
            buildup_fit (dict): Default empty dictionary for storing buildup fit results.
        """
        self.prefit = None
        self.single_fit = None
        self.global_fit = None
        self.buildup_fit = {}
