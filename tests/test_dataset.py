import CorziliusNMR.settings
from CorziliusNMR import dataset, settings
import unittest


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.ds = dataset.Dataset()

    def test_dataset_init_has_none_type_importer(self):
        self.assertIsNone(self.ds.importer)

    def test_dataset_init_properties(self):
        self.assertEqual(
            type(self.ds.props), CorziliusNMR.settings.Properties
        )

    def test_dataset_init_has_none_type_spectra(self):
        self.assertIsNone(self.ds.spectra)

    def test_dataset_init_has_none_type_fitter(self):
        self.assertIsNone(self.ds.fitter)

    def test_setup_correct_topspin_importer_default_properties(self):
        self.ds._setup_correct_topspin_importer()
        self.assertEqual(
            type(self.ds.importer), CorziliusNMR.io.Pseudo2DImporter
        )

    def test_setup_correct_topspin_importer_set_properties_pseudo2D(self):
        self.ds.props.expno = [2]
        self.ds._setup_correct_topspin_importer()
        self.assertEqual(
            type(self.ds.importer), CorziliusNMR.io.Pseudo2DImporter
        )

    def test_setup_correct_topspin_importer_set_properties_SCREAM(self):
        self.ds.props.expno = [2, 3, 4, 5, 6]
        self.ds._setup_correct_topspin_importer()
        self.assertEqual(
            type(self.ds.importer), CorziliusNMR.io.ScreamImporter
        )

    def test_read_in_data_from_topspin_pseudo2D(self):
        # TODO add some real parameters or fake spectrum
        self.ds._read_in_data_from_topspin
        self.assertIsNotNone(self.ds.spectra)
