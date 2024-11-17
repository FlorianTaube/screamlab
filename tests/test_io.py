import unittest

import CorziliusNMR.dataset
from CorziliusNMR import io
from CorziliusNMR.dataset import Dataset, _Experiment
class TestDataset(unittest.TestCase):

    def test_generate_output_csv_file_name(self):
        csvGenerator = io._FileNameHandler()
        csvGenerator.output_file_name = r"C:\TestFolder\Testfile"
        self.assertEqual(csvGenerator.generate_output_csv_file_name(), r"C:\TestFolder\Testfile.csv")

    def test_gen_export_pdf_file_name(self):
        pdf_generator = io._FileNameHandler()
        pdf_generator.output_file_name = r"C:\TestFolder\Testfile"
        self.assertEqual(pdf_generator.generate_export_output_pdf_file_name(),r"C:\TestFolder\Testfile.pdf")

    def test_if_topspin_scream_exporter_has_dataset(self):
        dataset = Dataset()
        screamExporter = io.ScreamExporter(dataset)
        self.assertEqual(type(screamExporter._dataset),CorziliusNMR.dataset.Dataset)

    def test_if_topspinpseudo_2D_exporter_has_dataset(self):
        dataset = Dataset()
        pseudoExporter = io.Pseudo2DExporter(dataset)
        self.assertEqual(type(pseudoExporter._dataset),CorziliusNMR.dataset.Dataset)

    def test_if_topspin_scream_exporter_has_dataset_with_correct_procpar_value(
            self):
        dataset = Dataset()
        dataset.procnoOfTopspinExperiment = "103"
        screamExporter = io.ScreamExporter(dataset)
        self.assertEqual(screamExporter._dataset.procno_of_topspin_experiment,
                         "103")

    def test_pathlist_to_experimental_data(self):
        pass

    def test_init_of_Experiment(self):
        exp = _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103")
        self.assertEqual(type(exp),CorziliusNMR._Experiment)

    def test_init_of_Experiment(self):
        exp = _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103")
        self.assertEqual(exp._file,r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103")

    def test_get_number_of_scans(self):
        exp = _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103")
        exp._get_values()
        self.assertEqual(exp.NS, 64)

    def test_get_x_axis(self):
        exp = _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103")
        exp._get_values()
        self.assertEqual(len(exp.x_axis), 16384)

    def test_get_y_axis(self):
        exp = _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103")
        exp._get_values()
        self.assertEqual(len(exp.y_axis), 16384)

    def test_get_normalize_y_data_to_number_of_scans(self):
        exp = _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103")
        exp._get_values()
        self.assertEqual(round(min(exp.y_axis)), -144)

