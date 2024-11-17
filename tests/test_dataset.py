import CorziliusNMR
from CorziliusNMR.dataset import Dataset, _Experiment
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

    def test_add_peak_to_all_exp(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict(sign="+",label="Test",
                                         fitting_group=1),
                             '50':dict(sign="+",label="Test2",fitting_group=2)}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(len(dataset.experiments[0].peaks),2)

    def test_assign_sign_when_sign_is_give(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict(sign="-",label="Test",
                                         fitting_group=1)}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(dataset.experiments[0].peaks[0].sign,"-")

    def test_assign_sign_when_parameter_not_minus_plus(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict(sign=12,label="Test",
                                         fitting_group=1)}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(dataset.experiments[0].peaks[0].sign,"+")

    def test_assign_sign_when_parameter_not_given(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict(label="Test",
                                         fitting_group=1)}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(dataset.experiments[0].peaks[0].sign,"+")

    def test_assign_peak_label_with_given_label(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict(label="Test",
                                         fitting_group=1)}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(dataset.experiments[0].peaks[0].peak_label,"Test")

    def test_assign_peak_label_without_given_label(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict(fitting_group=1)}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(dataset.experiments[0].peaks[0].peak_label,
                         "Peak_at_172_ppm")

    def test_assign_fitting_group(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict(fitting_group=1)}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(dataset.experiments[0].peaks[0].fitting_group,1)

    def test_assign_fitting_group_default(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict()}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(dataset.experiments[0].peaks[0].fitting_group,999)

    def test_assign_fitting_model_voigt(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict(fitting_model="voigt")}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(dataset.experiments[0].peaks[0].fitting_model, "voigt")

    def test_assign_fitting_model_voigt(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict(fitting_model="gauss")}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(dataset.experiments[0].peaks[0].fitting_model, "gauss")

    def test_assign_fitting_model_lorentz(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict(fitting_model="lorentz")}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(dataset.experiments[0].peaks[0].fitting_model,
                         "lorentz")

    def test_assign_fitting_model_default(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict()}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(dataset.experiments[0].peaks[0].fitting_model, "voigt")

    def test_assign_fitting_model_wrong_name(self):
        dataset = Dataset()
        dataset.experiments.append( _Experiment(
            r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103"))
        dataset.peak_dict = {'172': dict(fitting_model="Test")}
        dataset._add_peaks_to_all_exp()
        self.assertEqual(dataset.experiments[0].peaks[0].fitting_model, "voigt")

    def test_set_peak_hight(self):
        experiment = CorziliusNMR.dataset._Experiment(r"F:\NMR\Max\20230706_100mM_HN-P"
            r"-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K/24/pdata/103")
        experiment.x_axis = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
        experiment.y_axis = np.array([1,1,1, 1, 1, 2,3,2,1,1,1,1])
        peak = CorziliusNMR.dataset._Peak("5",{"5":dict(sign="+")},experiment)
        self.assertDictEqual(peak.hight,{'index':6,'x_val':6,'y_val':3})

