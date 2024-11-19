import os

import CorziliusNMR
from CorziliusNMR.dataset import Dataset, Spectra
import unittest
import numpy as np
import matplotlib.pyplot as plt

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset()
        self.dataset.output_file_name = "SCREAM_Test_Files/test_file"

    def fake_spectrum(self):
        # Define your spectrum
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].x_axis = np.linspace(0, 300, num=301)
        self.dataset.spectra[0].y_axis = \
            200000 * np.exp(-((self.dataset.spectra[0].x_axis - 150) ** 2) / (
                2 * 3 ** 2))
        self.dataset.peak_dict = {
            '149': dict(sign="+")
        }
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_hight()


    def test_set_path_to_experiment(self):
        self.dataset.path_to_topspin_experiment = "SCREAM_Test_Files/"
        self.assertEqual(self.dataset.path_to_topspin_experiment,
                         "SCREAM_Test_Files/")

    def test_set_path_to_output_folder(self):
        self.dataset.output_file_name = "SCREAM_Test_Files/tmp"
        self.assertEqual(self.dataset.output_file_name,
                         "SCREAM_Test_Files/tmp")

    def test_procno_of_topspin_experiment(self):
        self.dataset.procno_of_topspin_experiment = '103'
        self.assertEqual(self.dataset.procno_of_topspin_experiment,'103')

    def test_set_expno_of_topspin_experiment_1(self):
        self.dataset.expno_of_topspin_experiment = [22]
        self.assertEqual(self.dataset.expno_of_topspin_experiment, [22])

    def test_set_expno_of_topspin_experiment_2(self):
        self.dataset.expno_of_topspin_experiment = [22,24]
        self.assertTrue(np.array_equal(
            self.dataset.expno_of_topspin_experiment,np.array([22,23,24])))

    def test_set_expno_of_topspin_experiment_2(self):
        self.dataset.expno_of_topspin_experiment = [22,24,26]
        self.assertTrue(np.array_equal(
            self.dataset.expno_of_topspin_experiment,np.array([22,24,26])))

    def test_generate_output_csv_file_name(self):
        csv = self.dataset.file_name_generator.generate_output_csv_file_name()
        self.assertEqual(csv,"SCREAM_Test_Files/test_file.csv")

    def test_generate_output_pdf_file_name(self):
        csv = self.dataset.file_name_generator.generate_output_pdf_file_name()
        self.assertEqual(csv,"SCREAM_Test_Files/test_file.pdf")

    def test_setting_up_correct_topspin_importer_pseudo_2D(self):
        self.dataset.expno_of_topspin_experiment = [22]
        self.dataset._setup_correct_topspin_importer()
        self.assertEqual(type(self.dataset.importer),
                         CorziliusNMR.io.Pseudo2DImporter)

    def test_setting_up_correct_topspin_importer_scream(self):
        self.dataset.expno_of_topspin_experiment = [22,25]
        self.dataset._setup_correct_topspin_importer()
        self.assertEqual(type(self.dataset.importer),
                         CorziliusNMR.io.ScreamImporter)

    def test_setting_up_dataset_in_topspin_importer(self):
        self.dataset.expno_of_topspin_experiment = [22,25]
        self.dataset._setup_correct_topspin_importer()
        self.assertEqual(self.dataset,self.dataset.importer._dataset)


    def test_add_peak_2_peaks(self):
        self.dataset.peak_dict = {
            '172': dict(sign="+"),
            '50': dict(sign="-")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.assertEqual(len(self.dataset.spectra[0].peaks),2)

    def test_add_peak_set_sign_plus(self):
        self.dataset.peak_dict = {
            '172': dict(sign="+")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_sign()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].sign,"+")

    def test_add_peak_set_sign_minus(self):
        self.dataset.peak_dict = {
            '172': dict(sign="-")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_sign()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].sign,"-")

    def test_add_peak_set_sign_default(self):
        self.dataset.peak_dict = {
            '172': dict()
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_sign()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].sign,"+")
    def test_add_peak_set_sign_not_plusminus(self):
        self.dataset.peak_dict = {
            '172': dict(sign="hallo")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_sign()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].sign,"+")

    def test_add_peak_set_peak_label(self):
        self.dataset.peak_dict = {
            '172': dict(sign="hallo",label="Hallo")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_peak_label()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].peak_label,"Hallo")

    def test_add_peak_set_peak_default(self):
        self.dataset.peak_dict = {
            '172': dict(sign="hallo")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_peak_label()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].peak_label,"Peak_at_172_ppm")

    def test_add_peak_set_peak_default(self):
        self.dataset.peak_dict = {
            '172': dict(sign="hallo")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_fitting_group()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].fitting_group,999)

    def test_add_peak_set_peak_value(self):
        self.dataset.peak_dict = {
            '172': dict(sign="hallo",fitting_group=1)
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_fitting_group()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].fitting_group,1)

    def test_add_peak_set_peak_not_int(self):
        self.dataset.peak_dict = {
            '172': dict(sign="hallo",fitting_group="hal")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_fitting_group()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].fitting_group,999)


    def test_add_peak_set_fitting_model_wrong_input(self):
        self.dataset.peak_dict = {
            '172': dict(sign="hallo",fitting_model="hal")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_fitting_model()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].fitting_model,"voigt")

    def test_add_peak_set_fitting_model_voigt(self):
        self.dataset.peak_dict = {
            '172': dict(sign="hallo",fitting_model="voigt")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_fitting_model()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].fitting_model,"voigt")

    def test_add_peak_set_fitting_model_gauss(self):
        self.dataset.peak_dict = {
            '172': dict(sign="hallo",fitting_model="gauss")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_fitting_model()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].fitting_model,"gauss")

    def test_add_peak_set_fitting_model_lorentz(self):
        self.dataset.peak_dict = {
            '172': dict(sign="hallo",fitting_model="lorentz")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_fitting_model()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].fitting_model,"lorentz")
    def test_add_peak_set_fitting_model_nan(self):
        self.dataset.peak_dict = {
            '172': dict(sign="hallo")
        }
        self.dataset.spectra.append(Spectra(self.dataset))
        self.dataset.spectra[0].add_peak(self.dataset.peak_dict)
        self.dataset.spectra[0].peaks[0]._set_fitting_model()
        self.assertEqual(self.dataset.spectra[0].peaks[
                              0].fitting_model,"voigt")

    def test_set_hight_x_val(self):
        self.fake_spectrum()
        self.assertEqual(self.dataset.spectra[0].peaks[0].hight['x_val'],150)

    def test_set_hight_y_val(self):
        self.fake_spectrum()
        self.assertEqual(self.dataset.spectra[0].peaks[0].hight['y_val'],200000)

    def test_set_hight_index(self):
        self.fake_spectrum()
        self.assertEqual(self.dataset.spectra[0].peaks[0].hight['index'],150)

