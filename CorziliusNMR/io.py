"""
io module of the CorziliusNMR package.
"""
import CorziliusNMR.dataset
import numpy as np
from bruker.data.nmr import *
import bruker.api.topspin as top
import matplotlib.pyplot as plt
import os

class TopspinImporter:

    def __init__(self,dataset):
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
        self._dataset.spectra[-1].x_axis = np.linspace(float(_physicalRange[
                                                             'start']),
                                                       float(_physicalRange[ 'end']),
                                                       _number_of_datapoints)

    def _set_y_axis(self):
        self._dataset.spectra[-1].y_axis = \
            self._nmr_data.getSpecDataPoints()['dataPoints']

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
        #self.print_fit_per_fitting_spectrum()
        #self.print_fitting_report()
        #self.print_table_with_fitting_results()
        #self.print_biexp_fits()
        #self.print_biexp_table()

    def print_pdf_from_import(self):
        for spectrum in self.dataset.spectra:
            plt.plot(spectrum.x_axis,spectrum.y_axis, label=f"$t_{{del}}$ = "
                                                            f"{spectrum.tbup} s")
        plt.legend()
        plt.xlabel("$chemical\ shift$ / ppm", fontsize=14)
        plt.ylabel("$scan\ normalized\ signal\ intensity$ / a.u.", fontsize=14)
        plt.gca().invert_xaxis()  # Invert the x-axis
        plt.tight_layout()
        file = self.dataset.file_name_generator.ge
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
                    plt.plot(spectrum.x_axis,spectrum.y_axis,color="black",
                             label="Experiment")
                    sim = 0
                    for peak in spectrum.peaks:
                        plt.plot(spectrum.x_axis,peak.simulated_peak[
                            fitting_type],"b:",alpha=0.9)
                        sim += peak.simulated_peak[
                            fitting_type]
                    plt.plot(spectrum.x_axis,sim,"r--",label="Simulation")
                    plt.xlabel("$chemical\ shift$ / ppm", fontsize=14)
                    plt.ylabel("$scan\ normalized\ signal\ intensity$ / a.u.",
                               fontsize=14)
                    plt.gca().invert_xaxis()  # Invert the x-axis
                    plt.tight_layout()
                    plt.legend()
                    plt.savefig(self.dataset.file_name_generator
                                .generate_spectrum_fit_pdf(fitting_type,
                                                           spectrum.tbup),
                                format="pdf",
                                bbox_inches="tight")  # Save as PDF







