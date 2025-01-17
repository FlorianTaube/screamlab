import unittest
from CorziliusNMR.settings import Properties


class TestProperties(unittest.TestCase):

    def test_prefit_default_value(self):
        props = Properties()
        self.assertFalse(props.prefit)

    def test_prefit_initial_value(self):
        props = Properties(prefit=True)
        self.assertTrue(props.prefit)

    def test_prefit_set_valid_value(self):
        props = Properties()
        props.prefit = True
        self.assertTrue(props.prefit)

    def test_prefit_set_invalid_value(self):
        props = Properties()
        with self.assertRaises(TypeError) as context:
            props.prefit = "invalid"
        self.assertEqual(
            str(context.exception),
            "Expected 'prefit' to be of type 'bool', got str.",
        )

    def test_prefit_change_value(self):
        props = Properties(prefit=False)
        props.prefit = True
        self.assertTrue(props.prefit)

    def test_prefit_private_variable(self):
        props = Properties(prefit=True)
        self.assertTrue(props._prefit)

    def test_buildup_types_default_value(self):
        props = Properties()
        self.assertListEqual(props.buildup_types, ["exponential"])

    def test_buildup_types_initial_value(self):
        props = Properties(buildup_types=["exponential"])
        self.assertListEqual(props.buildup_types, ["exponential"])

    def test_buildup_types_set_valid_value(self):
        props = Properties()
        props.buildup_types = ["exponential", "biexponential"]
        self.assertListEqual(
            props.buildup_types, ["exponential", "biexponential"]
        )

    def test_buildup_types_set_invalid_value(self):
        props = Properties()
        with self.assertRaises(TypeError) as context:
            props.buildup_types = "invalid"
        self.assertEqual(
            str(context.exception),
            "Expected 'buildup_types' to be of type 'list', got str.",
        )

    def test_buildup_types_change_value(self):
        props = Properties(buildup_types=["biexponential"])
        props.buildup_types = ["exponential", "biexponential"]
        self.assertListEqual(
            props.buildup_types, ["exponential", "biexponential"]
        )

    def test_buildup_types_private_variable(self):
        props = Properties(buildup_types=["biexponential"])
        self.assertListEqual(props._buildup_types, ["biexponential"])

    def test_buildup_types_empty_buildup_types(self):
        props = Properties()
        empty_list = []
        props.buildup_types = empty_list
        self.assertEqual(props.buildup_types, empty_list)

    def test_buildup_types_invalid_buildup_types_contains_non_strings(self):
        props = Properties()
        with self.assertRaises(ValueError):
            props.buildup_types = ["exponential", 42, "constant"]

    def test_buildup_types_valid_in_possibilitys(self):
        props = Properties()
        try:
            props.buildup_types = ["exponential", "biexponential"]
            self.assertEqual(
                props.buildup_types, ["exponential", "biexponential"]
            )
        except Exception as e:
            self.fail(f"Valid values raised an exception: {e}")

    def test_buildup_types_invalid(self):
        props = Properties()
        with self.assertRaises(ValueError) as context:
            props.buildup_types = ["exponential", "invalid_type"]
        self.assertIn("must be one of", str(context.exception))

    def test_spectrum_for_prefit_default_value(self):
        props = Properties()
        self.assertEqual(props.spectrum_for_prefit, 0)

    def test_spectrum_for_prefit_initial_value(self):
        props = Properties(spectrum_for_prefit=1)
        self.assertEqual(props.spectrum_for_prefit, 1)

    def test_spectrum_for_prefit_set_valid_value(self):
        props = Properties()
        props.spectrum_for_prefit = 2
        self.assertEqual(props.spectrum_for_prefit, 2)

    def test_spectrum_for_prefit_set_invalid_value(self):
        props = Properties()
        with self.assertRaises(TypeError) as context:
            props.spectrum_for_prefit = "invalid"
        self.assertEqual(
            str(context.exception),
            "Expected 'spectrum_for_prefit' to be of type 'int', got str.",
        )

    def test_spectrum_for_prefit_change_value(self):
        props = Properties(spectrum_for_prefit=1)
        props.spectrum_for_prefit = 2
        self.assertEqual(props.spectrum_for_prefit, 2)

    def test_spectrum_for_prefitt_private_variable(self):
        props = Properties(spectrum_for_prefit=2)
        self.assertTrue(props._spectrum_for_prefit)
