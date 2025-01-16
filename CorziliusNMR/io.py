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
import sys
from tabulate import tabulate

class TopspinImporter:

    def __init__(self,dataset):
        self._dataset = dataset
        self._top = top.Topspin()
        self._data_provider = self._top.getDataProvider()
        self._current_path_to_exp = None
        self._nmr_data = None
        self._len = 0

    def import_topspin_data(self):
        pass

    def _set_values(self):
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

class ScreamImporter(TopspinImporter):

    def __init__(self,dataset):
        super().__init__(dataset)

    def import_topspin_data(self):
        path = self._dataset.file_name_generator.path_to_topspin_experiment
        procno =self._dataset.file_name_generator.procno_of_topspin_experiment
        for expno_nr,expno in \
                enumerate(self._dataset.file_name_generator.expno_of_topspin_experiment):
            file = os.path.join(path, str(expno), "pdata", str(procno))
            self._dataset.spectra.append(CorziliusNMR.dataset.Spectra(
                file))
            self._nmr_data = self._data_provider.getNMRData(file)
            self._set_values()

    def _set_values(self):
        super()._set_values()

    def _set_number_of_scans(self):
        self._dataset.spectra[-1].NS = int(self._nmr_data.getPar("NS"))

    def _set_buildup_time(self):
        self._dataset.spectra[-1].tbup = int(self._nmr_data.getPar("L 20")) / 4


    def _set_x_data(self):
        _physicalRange = self._nmr_data.getSpecDataPoints()['physicalRanges'][0]
        _number_of_datapoints = len(self._nmr_data.getSpecDataPoints()[
            'dataPoints'])
        axis = np.linspace(float(_physicalRange['start']), float(_physicalRange[ 'end']),
                                                       _number_of_datapoints)
        self._dataset.spectra[-1].x_axis = \
            utils.generate_subspectrum_2(axis,axis,
                    max(list(map(int, self._dataset.peak_dict.keys()))),
                    min(list(map(int, self._dataset.peak_dict.keys()))),50)
        if self._len == 0:
            self._len = len(self._dataset.spectra[-1].x_axis)
        else:
            if self._len > len(self._dataset.spectra[-1].x_axis):
                diff = self._len - len(self._dataset.spectra[-1].x_axis)
                count = 0
                while count < diff:
                    self._dataset.spectra[-1].x_axis = np.append(self._dataset.spectra[-1].x_axis,0)
                    count += 1

    def _set_y_axis(self):
        axis = self._nmr_data.getSpecDataPoints()['dataPoints']
        _physicalRange = self._nmr_data.getSpecDataPoints()['physicalRanges'][0]
        _number_of_datapoints = len(self._nmr_data.getSpecDataPoints()[
                                        'dataPoints'])
        x_axis = np.linspace(float(_physicalRange['start']), float(
            _physicalRange[ 'end']), _number_of_datapoints)
        self._dataset.spectra[-1].y_axis = \
            utils.generate_subspectrum_2(x_axis,axis,
                    max(list(map(int, self._dataset.peak_dict.keys()))),
                    min(list(map(int, self._dataset.peak_dict.keys()))), 50)
        if self._len == 0:
            self._len = len(self._dataset.spectra[-1].y_axis)
        else:
            if self._len > len(self._dataset.spectra[-1].y_axis):
                diff = self._len - len(self._dataset.spectra[-1].y_axis)
                count = 0
                while count < diff:
                    self._dataset.spectra[-1].y_axis = np.append(self._dataset.spectra[-1].y_axis,0)
                    count += 1

    def _normalize_y_values_to_number_of_scans(self):
        self._dataset.spectra[-1].y_axis = np.divide(
            self._dataset.spectra[-1].y_axis, self._dataset.spectra[
                -1].NS)

class Pseudo2DImporter(TopspinImporter):

    def __init__(self,dataset):
        super().__init__(dataset)

    def import_topspin_data(self):
        return

    def read_in_topspin_data():
        pass


class Exporter:

    def __init__(self,dataset):
        self.dataset = dataset
        pass

    def print_all(self):
        self.print_csv_from_import()
        self.print_pdf_from_import()
        if self.dataset._print_each_peak_fit_seperate:
            self.print_fit_per_fitting_spectrum()

        if self.dataset._print_complete_fit_report:
            self.print_fitting_report()
        self.print_summary_as_txt()
        for fitting_type in self.dataset.buildup_type:
            self.print_biexp_fits(fitting_type)
            self.print_biexp_table(fitting_type)
        self.print_all_fits_in_one()

    def print_pdf_from_import(self):
        for spectrum in self.dataset.spectra:
            plt.plot(spectrum.x_axis,spectrum.y_axis, label=f"$t_{{del}}$ = "
                                                            f"{spectrum.tbup} s")
        plt.legend()
        plt.xlabel("$chemical\ shift$ / ppm", fontsize=12)
        plt.ylabel("$scan\ normalized\ signal\ intensity$ / a.u.", fontsize=12)
        plt.gca().invert_xaxis()  # Invert the x-x_axis
        plt.tight_layout()
        file = self.dataset.file_name_generator.generate_output_pdf_file_name()
        plt.savefig(file, format="pdf", bbox_inches="tight")  # Save as PDF
        #plt.show()
        plt.close()

    def print_csv_from_import(self):
        output = None
        delay_times = []
        for spectrum_nr,spectrum in enumerate(self.dataset.spectra):
            delay_times.append(str(spectrum.tbup))
            delay_times.append("")
            if spectrum_nr == 0:
                output = spectrum.x_axis
            else:
                output = np.vstack((output,spectrum.x_axis))
            output = np.vstack((output, spectrum.y_axis))
        np.savetxt(self.dataset.file_name_generator
        .generate_output_csv_file_name(), output.transpose(), delimiter=";",
                   header=";".join(delay_times))

    def print_fitting_report(self):
        for fitting_type in self.dataset.spectrum_fitting_type:
            if fitting_type == "global":
                for peak in self.dataset.spectra[0].peaks:
                    file = \
                        self.dataset.file_name_generator.\
                            generate_txt_fitting_report\
                            ("_".join(peak.peak_label.split("_")[0:4]),fitting_type)
                    with open(file, "w") as txt_file:
                        txt_file.write(str(peak.fitting_report[fitting_type]))

    def print_fit_per_fitting_spectrum(self):
        for fitting_type in self.dataset.spectrum_fitting_type:
            if fitting_type == "global":
                for spectrum in self.dataset.spectra:
                    fig = plt.figure(figsize=(8, 6))
                    gs = GridSpec(2, 1, height_ratios=[3, 1],
                                  hspace=0.1)
                    ax1 = fig.add_subplot(gs[0])
                    ax1.plot(spectrum.x_axis, spectrum.y_axis, color="black",
                             label="Experiment")
                    sim = 0
                    for peak in spectrum.peaks:
                        ax1.plot(spectrum.x_axis,
                                 peak.simulated_peak[fitting_type], "b:",
                                 alpha=0.9)
                        sim += peak.simulated_peak[fitting_type]
                    ax1.plot(spectrum.x_axis, sim, "r--", label="Simulation")
                    ax1.set_xlabel("")
                    ax1.set_ylabel(
                        "$scan\ normalized\ signal\ intensity$ / a.u.",
                        fontsize=13)
                    ax1.legend()
                    ax1.invert_xaxis()
                    ax2 = fig.add_subplot(gs[1],
                                          sharex=ax1)
                    residual = sim - spectrum.y_axis
                    ax2.plot(spectrum.x_axis, residual, color="#888888",
                             label="Residual")
                    ax2.axhline(0, color="black", linestyle="--",
                                linewidth=0.8)
                    ax2.set_xlabel("$chemical\ shift$ / ppm", fontsize=13)
                    ax2.set_ylabel("$Residual$", fontsize=13)
                    ax2.invert_xaxis()
                    maximum = max(abs(spectrum.y_axis))
                    ax2.set_ylim(-maximum/3,maximum/3)
                    plt.setp(ax1.get_xticklabels(), visible=False)
                    plt.savefig(
                    self.dataset.file_name_generator.generate_spectrum_fit_pdf(
                            fitting_type, spectrum.tbup), format="pdf",
                        bbox_inches="tight")
                    plt.close()

    def print_all_fits_in_one(self):
        plt.close()
        for fitting_type in self.dataset.spectrum_fitting_type:
            if fitting_type == "global":
                for spectrum in self.dataset.spectra:
                    plt.plot(spectrum.x_axis, spectrum.y_axis, color="black")
                    sim = 0
                    for peak in spectrum.peaks:
                        #plt.plot(spectrum.x_axis,
                         #        peak.simulated_peak[fitting_type], "r--",
                          #       alpha=0.9)
                        sim += peak.simulated_peak[fitting_type]
                    plt.plot(spectrum.x_axis, sim, "r--", linewidth=0.7, alpha=0.75)
                ax = plt.gca()
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                plt.xlabel(r"$\delta (^{13}\text{C})$ / ppm", fontsize=13)
                plt.ylabel("$I$ / a.u.", fontsize=13)
                plt.legend(["Experiment", "Simulation"])
                plt.gca().invert_xaxis()
                plt.savefig(
                    self.dataset.file_name_generator.generate_all_spectrum_fit_pdf(
                        fitting_type), format="pdf",
                    bbox_inches="tight"
                )
                plt.close()

    def print_biexp_fits(self,fitting_type):
        colors = plt.cm.Set2.colors  # Example: 10 colors from the tab10 colormap
        for idx, (peak_label, peak_data) in enumerate(
                self.dataset._exp_fit[fitting_type].items()):
            color = colors[idx % len(colors)]  # Cycle through the colors
            plt.plot(peak_data['x_axis'], peak_data['y_axis'], "o",
                     color=color,label=peak_label)
            tau = np.linspace(0, max(peak_data['x_axis']), 1000)
            result = peak_data['result']
            sim = result.eval(x=tau)
            plt.plot(tau, sim, color=color)
        plt.xlabel("$delay\ time$ / s",fontsize=14)
        plt.ylabel("$signal\ intensity$ \ a.u.",fontsize=14)
        plt.legend()
        plt.tight_layout()
        #plt.xlim([0,20])
        file = self.dataset.file_name_generator.generate_buildup_pdf(fitting_type)
        plt.savefig(file, format="pdf", bbox_inches="tight")
        plt.close()


    def print_biexp_table(self, fitting_type):
        table = []
        header = ["Peak"]

        for idx, (peak_label, peak_data) in enumerate(
                self.dataset._exp_fit[fitting_type].items()):
            for name in peak_data['params'].keys():
                if name not in header:
                    header.append(name)
        for peak_label, peak_data in self.dataset._exp_fit[
            fitting_type].items():
            row = [peak_label]
            for name in header[1:]:
                param_value = peak_data['params'].get(name, None)
                row.append(
                    param_value.value if param_value is not None else "N/A")

            table.append(row)
        table_output = tabulate(table, headers=header, tablefmt="grid", stralign="center")
        with open(self.dataset.file_name_generator.generate_buildup_txt(fitting_type),
                  "w") as file:
            file.write(table_output)

    def print_summary_as_txt(self):
        time_list = defaultdict(list)
        amp_list = defaultdict(list)
        integral = defaultdict(list)
        for spectrum in self.dataset.spectra:
            cen_list = defaultdict(list)
            sig_list = defaultdict(list)
            gam_list = defaultdict(list)
            for peak in spectrum.peaks:
                key = " ".join(peak.peak_label.split("_")[0:3])
                time = " ".join(peak.peak_label.split("_")[4:-1])
                time_list[key].append(time)
                integral[key].append(peak.area_under_peak["global"])
                for param_name, param_value in peak.fitting_parameter[
                    'global'].items():
                    if param_name == "amp":
                        amp_list[key].append(param_value.value)
                    if param_name == "cen":
                        cen_list[key].append(param_value.value)
                    if param_name == "gam":
                        gam_list[key].append(param_value.value)
                    if param_name == "sig":
                        sig_list[key].append(param_value.value)
        lorentz = defaultdict(list)
        gauss = defaultdict(list)
        voigt = defaultdict(list)
        for keys in sig_list:
            lorentz[keys] = utils.fwhm_lorentzian(gam_list[keys][0])
            gauss[keys] = utils.fwhm_gaussian(sig_list[keys][0])
            voigt[keys] = utils.fwhm_voigt(sig_list[keys][0],gam_list[keys][0])

        with open(self.dataset.file_name_generator.generate_summary_txt(),
                "w") as txt_file:
            txt_file.write(" ;cen;sig;gam")
            for time in time_list[key]:
                txt_file.write(f";amp({time})")
            txt_file.write(" ;FWHM Lorentzian;FWHM Gaussian;FWHM Voigt")
            for time in time_list[key]:
                txt_file.write(f";integral({time})")
            txt_file.write("\n")
            for keys in cen_list:
                txt_file.write(f"{keys};{cen_list[keys][0]};{sig_list[keys][0]};"
                               f"{gam_list[keys][0]}")
                for amp in amp_list[key]:
                    txt_file.write(f";{amp}")
                txt_file.write(f";{lorentz[keys]};{gauss[keys]};{voigt[keys]}")
                for int in integral[key]:
                    txt_file.write(f";{int}")
                txt_file.write("\n")
            txt_file.write("\n")
            txt_file.write("\n")



        pass







