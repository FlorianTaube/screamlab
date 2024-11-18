import unittest

import CorziliusNMR.dataset
from CorziliusNMR import io
from CorziliusNMR.dataset import Dataset, Experiment
class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset()


    def tearDown(self):
        #os.rmdir('tmp/')
        pass
    def fake_input(self):
        # Define your spectrum
        self.dataset.output_file_name = "../tests/SCREAM_Test_Files/tmp/"
        self.dataset.path_to_topspin_experiment = r"..\tests\SCREAM_Test_Files"
        self.dataset.expno_of_topspin_experiment = [24, 26]
        self.dataset.procno_of_topspin_experiment = "103"

    def test_import_topspin_data(self):
        self.fake_input()
        self.dataset._setup_correct_topspin_importer()
        self.dataset.importer.import_topspin_data()
        self.assertEqual(len(self.dataset.experiments),3)

    def test_import_of_set_nmr_data(self):
        self.fake_input()
        self.dataset._setup_correct_topspin_importer()
        self.dataset.importer.import_topspin_data()
        self.assertEqual(len(self.dataset.experiments),3)

