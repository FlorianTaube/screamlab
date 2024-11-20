import unittest

import CorziliusNMR.dataset


class TestDataset(unittest.TestCase):

    def first_dataset(self):
        print("hallo")
        dataset = CorziliusNMR.dataset.Dataset()
        dataset.expno_of_topspin_experiment =[28,33]
        dataset.procno_of_topspin_experiment = "103"
        dataset.output_file_name = "../tests/SCREAM_Test_Files/tmp/"
        dataset.start_buildup_fit_from_topspin_export()


