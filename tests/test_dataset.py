import CorziliusNMR
from CorziliusNMR.dataset import Dataset
import unittest
import numpy as np

class TestDataset(unittest.TestCase):
    def test_set_path_to_experiment(self):
        dataset = Dataset()
        path = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
        dataset.path_to_topspin_experiment= path
        self.assertEqual(dataset._fileNames.path_to_topspin_experiment, path)

    def test_set_expno_to_experiment_one_value(self):
        dataset = Dataset()
        dataset.expno_of_topspin_experiment = [24]
        self.assertTrue(np.array_equal([24], dataset.expno_of_topspin_experiment))

    def test_set_expno_to_experiment_more_values(self):
        dataset = Dataset()
        dataset.expno_of_topspin_experiment = [24, 128, 56, 1]
        self.assertTrue(np.array_equal([24,128,56,1], dataset.expno_of_topspin_experiment))

    def test_set_expno_to_experiment_from_to(self):
        dataset = Dataset()
        dataset.expno_of_topspin_experiment = [24, 33]
        self.assertTrue(np.array_equal(np.arange(24,34),
                                       dataset.expno_of_topspin_experiment))

    def test_set_procno_to_experiment(self):
        dataset = Dataset()
        dataset.procnoOfTopspinExperiment = 103
        self.assertEqual(dataset._fileNames.procno_of_topspin_experiment, str(103))

    def test_set_output_file_to_experiment(self):
        dataset = Dataset()
        file = r"C:\Users\Florian Taube\Desktop\Prolin_auswertung_Test\HN-P-100K"
        dataset.output_file_name = file
        self.assertEqual(dataset._fileNames.output_file_name, file)

    def test_setup_correct_topspin_exporter_oseudo2D(self):
        dataset = Dataset()
        dataset.expno_of_topspin_experiment = [22]
        dataset._setup_correct_topspin_exporter()
        self.assertEqual(type(dataset._topspin_exporter), CorziliusNMR.io.Pseudo2DExporter)

    def test_setup_correct_topspin_exporter_scream(self):
        dataset = Dataset()
        dataset.expno_of_topspin_experiment = [22, 34]
        dataset._setup_correct_topspin_exporter()
        self.assertEqual(type(dataset._topspin_exporter),
                         CorziliusNMR.io.ScreamExporter)


    def test_pathlist_to_experimental_data(self):
        dataset = Dataset()
        dataset.procnoOfTopspinExperiment = 103
        dataset.expno_of_topspin_experiment = [24, 33]
        dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
        dataset.start_buildup_fit_from_topspin_export()
        self.assertEqual(len(dataset._topspin_exporter.experiments), 10)

    def test_add_experimental_data_to_dataset(self):
        dataset = Dataset()
        dataset.procnoOfTopspinExperiment = 103
        dataset.expno_of_topspin_experiment = [24, 33]
        dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
        dataset.output_file_name = r"C:\Users\Florian " \
                                    r"Taube\Desktop\Prolin_auswertung_Test\HN-P-100K"
        dataset.start_buildup_fit_from_topspin_export()
        self.assertEqual(len(dataset.experiments), 10)



    def test_add_experimental_data_to_dataset_three_spectra(self):
        dataset = Dataset()
        dataset.procnoOfTopspinExperiment = 103
        dataset.expno_of_topspin_experiment = [30, 33]
        dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
        dataset.output_file_name = r"C:\Users\Florian " \
                                    r"Taube\Desktop\Prolin_auswertung_Test\HN-P-100K"
        dataset.start_buildup_fit_from_topspin_export()
        self.assertEqual(len(dataset.experiments), 4)

