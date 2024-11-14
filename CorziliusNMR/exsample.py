from CorziliusNMR.dataset import Dataset

dataset = Dataset()
dataset.procnoOfTopspinExperiment = 103
dataset.expno_of_topspin_experiment = [24, 33]
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
print(dataset.path_to_topspin_experiment)
print(dataset.expno_of_topspin_experiment)
print(dataset.procnoOfTopspinExperiment)
dataset.start_buildup_fit_from_topspin_export()

