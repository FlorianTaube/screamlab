import unittest

import CorziliusNMR.dataset
from CorziliusNMR import io
from CorziliusNMR.dataset import Dataset, Spectra
class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset()


    def tearDown(self):
        #os.rmdir('tmp/')
        pass
    def fake_input(self):
        # Define your spectrum
        self.dataset.output_file_name = "../tests/SCREAM_Test_Files/tmp/"
        self.dataset.path_to_topspin_experiment = r"C:\Users\Florian " \
                r"Taube\Documents\Programmierung\CorziliusNMR\tests\SCREAM_Test_Files"
        self.dataset.expno_of_topspin_experiment = [24, 26]
        self.dataset.procno_of_topspin_experiment = "103"

    def fake_dataset_for_scream_input(self):
        self.fake_input()
        file =r"C:\Users\Florian " \
              r"Taube\Documents\Programmierung\CorziliusNMR\tests" \
              r"\SCREAM_Test_Files\24\pdata\103"
        self.dataset._setup_correct_topspin_importer()
        self.dataset.importer.import_topspin_data()

    def test_import_topspin_data(self):
        self.fake_input()
        self.dataset._setup_correct_topspin_importer()
        self.dataset.importer.import_topspin_data()
        self.assertEqual(len(self.dataset.spectra), 3)


    def test_import_topspin_data_and_set_file_name(self):
        self.fake_input()
        self.dataset._setup_correct_topspin_importer()
        self.dataset.importer.import_topspin_data()
        self.assertEqual(self.dataset.spectra[0].file, r"C:\Users\Florian "
                         r"Taube\Documents\Programmierung\CorziliusNMR\tests"
                          r"\SCREAM_Test_Files\24\pdata\103")

    def test_import_of_set_number_of_scans_1(self):
        self.fake_dataset_for_scream_input()
        self.assertEqual(self.dataset.spectra[0].NS, 64)

    def test_import_of_set_number_of_scans_2(self):
        self.fake_dataset_for_scream_input()
        self.assertEqual(self.dataset.spectra[2].NS, 32)

    def test_import_of_set_buildup_1(self):
        self.fake_dataset_for_scream_input()
        self.assertEqual(self.dataset.spectra[0].tbup, 2)

    def test_import_of_set_x_axis(self):
        self.fake_dataset_for_scream_input()
        self.assertEqual(len(self.dataset.spectra[0].x_axis), 16384)

    def test_import_of_set_y_axis(self):
        self.fake_dataset_for_scream_input()
        self.assertEqual(len(self.dataset.spectra[0].y_axis), 16384)

    def test_import_of_set_normalize_y_value(self):
        self.fake_dataset_for_scream_input()
        self.assertEqual(int(max(self.dataset.spectra[0].y_axis)), 673)

