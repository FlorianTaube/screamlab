import unittest

from CorziliusNMR import io,dataset
class TestDataset(unittest.TestCase):

    def setUp(self):
        self.scream_importer = io.ScreamImporter(dataset.Dataset)
        self.pseudo_importer = io.Pseudo2DImporter(dataset.Dataset)


    def test_scream_init_set_dataset(self):
        self.assertEqual(type(self.scream_importer._dataset),CorziliusNMR.)

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

