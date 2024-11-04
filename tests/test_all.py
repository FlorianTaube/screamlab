import sys
from CorziliusNMR import io, fitting, SCREAM_mod_analyzer
import unittest

class TestAnalysis(unittest.TestCase):
    def test_S27F(self):
        path_of_experiments = "F:/ssNMR/20241021_S27F/"
        tb1H_list = [ 0]
        for tb_nr,tb in enumerate(tb1H_list):
            expno = [161,168]
            procno = "103"

            output_file = "C:/Users/Florian " \
                      "Taube/Desktop/SCREAM_Project/Ala-AMUPOL_60_30_10_" \
                          ""+str(tb)
            io.generate_csv_from_scream_set(path_of_experiments,
                                        output_file, expno)
            print("hello")
            sys.exit()
            peak_center = [-16]
            peaks_for_single_fit = [1]
            plus_minus = ["+"]
            #fitting.start_voigt_fitting(output_file,peak_center,
            #                        peaks_for_single_fit=peaks_for_single_fit,
            #                       autopeakpick=True, global_fit=True)
        output_file = "C:/Users/Florian " \
                      "Taube/Desktop/SCREAM_Project/Ala-AMUPOL_60_30_10_"
        SCREAM_mod_analyzer.plot_scream_mod(output_file, sorted(tb1H_list))

    def test_S28F(self):
        path_of_experiments = "F:/ssNMR/20241022_S28F/"
        tb1H_list = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 0, 4]
        for tb_nr,tb in enumerate(tb1H_list):
            expno = [1+tb_nr*20,7+tb_nr*20]
            procno = "103"

            output_file = "C:/Users/Florian " \
                      "Taube/Desktop/SCREAM_Project/Ala-AMUPOL_60_10_30_" \
                          ""+str(tb)
            #io.generate_csv_from_scream_set(path_of_experiments,
            #                            output_file, expno)

            peak_center = [-16]
            peaks_for_single_fit = [1]
            plus_minus = ["+"]
            #fitting.start_voigt_fitting(output_file,peak_center,
            #                        peaks_for_single_fit=peaks_for_single_fit,
            #                       autopeakpick=True, global_fit=True)
        output_file = "C:/Users/Florian " \
                      "Taube/Desktop/SCREAM_Project/Ala-AMUPOL_60_10_30_"
        SCREAM_mod_analyzer.plot_scream_mod(output_file, sorted(tb1H_list))
    def test_S29F(self):
        path_of_experiments = "F:/ssNMR/20241023_S29F/"
        tb1H_list = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 0, 4]

        for tb_nr,tb in enumerate(tb1H_list):
            expno = [1+tb_nr*20,7+tb_nr*20]
            procno = "103"

            output_file = "C:/Users/Florian " \
                      "Taube/Desktop/SCREAM_Project/Ala-Asympol_60_30_10_" \
                          ""+str(tb)
            #io.generate_csv_from_scream_set(path_of_experiments,
            #                            output_file, expno)

            peak_center = [-16]
            peaks_for_single_fit = [1]
            plus_minus = ["+"]
            #fitting.start_voigt_fitting(output_file,peak_center,
            #                        peaks_for_single_fit=peaks_for_single_fit,
            #                       autopeakpick=True, global_fit=True)
        output_file = "C:/Users/Florian " \
                      "Taube/Desktop/SCREAM_Project/Ala-Asympol_60_30_10_"
        SCREAM_mod_analyzer.plot_scream_mod(output_file, sorted(tb1H_list))

    def test_prolin(self):
        path_of_experiments ="F:/NMR/Max/20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/"
        expno = [24,33]
        procno = "103"

        output_file = "C:/Users/Florian Taube/Desktop/Pro_Test"

        io.generate_csv_from_scream_set(path_of_experiments,
                                    output_file, expno)
