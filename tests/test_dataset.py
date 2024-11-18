import os

import CorziliusNMR
from CorziliusNMR.dataset import Dataset, Experiment
import unittest
import numpy as np

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset()
        self.dataset.output_file_name = "SCREAM_Test_Files/test_file"

    def tearDown(self):
        #os.rmdir('tmp/')
        pass
    def fake_spectrum(self):
        # Define your spectrum
        self.dataset = Dataset()
        self.dataset.experiments[0].x_axis = np.linspace(0,300, num=301)

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



