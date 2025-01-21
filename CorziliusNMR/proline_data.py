from CorziliusNMR import dataset as ds
import sys

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20250120_S47F_100K"
dataset.expno_of_topspin_experiment = [24, 32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Ala-Prolin_100K_13mm"
)
dataset.peak_dict = {
    "160": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "42": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "30": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "11": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "8": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

sys.exit()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
dataset.expno_of_topspin_experiment = [24, 33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin_100K_13mm"
)
dataset.peak_dict = {
    "175": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    #'61': dict(sign="+", fitting_group=1, fitting_model="voigt"),
    #'46': dict(sign="+", fitting_group=1, fitting_model="voigt"),
    #'30': dict(sign="+", fitting_group=1, fitting_model="voigt"),
    #'24': dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

sys.exit()


dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241126_S33F_100K"
dataset.expno_of_topspin_experiment = [23, 32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin_100K_32mm"
)
dataset.peak_dict = {
    "175": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "46": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "30": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "24": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -2
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241126_S33F_125K"
dataset.expno_of_topspin_experiment = [23, 32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin_125K_32mm"
)
dataset.peak_dict = {
    "175": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "46": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "30": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "24": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -2
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241126_S34F_150K"
dataset.expno_of_topspin_experiment = [23, 32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin_150K_32mm"
)
dataset.peak_dict = {
    "175": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "46": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "30": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "24": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -2
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()


dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241128_S36F_100K"
dataset.expno_of_topspin_experiment = [23, 32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin_na_100K_32mm"
)
dataset.peak_dict = {
    #'175': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "46": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "30": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "24": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241129_S36F_125K"
dataset.expno_of_topspin_experiment = [23, 32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin_na_125K_32mm"
)
dataset.peak_dict = {
    "175": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "46": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "30": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "24": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241203_S36F_150K"
dataset.expno_of_topspin_experiment = [23, 32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin_na_150K_32mm"
)
dataset.peak_dict = {
    "175": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "46": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "30": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "24": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -5
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()


dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20230816_1M_HN-P-OH_na_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
)
dataset.expno_of_topspin_experiment = [24, 32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin_na_150K_13mm"
)
dataset.peak_dict = {
    "175": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "46": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "30": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "24": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -3
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()


dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_125K"
)
dataset.expno_of_topspin_experiment = [24, 33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin_125K_13mm"
)
dataset.peak_dict = {
    "175": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "46": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "30": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "24": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
)
dataset.expno_of_topspin_experiment = [24, 33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin_150K_13mm"
)
dataset.peak_dict = {
    "175": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "46": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "30": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "24": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20230620_100mM_H2N-PG-OH_10mM_AMUPOL_1p3mm_18kHz_DNP_100K"
)
dataset.expno_of_topspin_experiment = [24, 33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin-Glycin_100K_13mm"
dataset.peak_dict = {
    "171": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "48": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "32": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "25": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20230814_100mM_H2N-PG-OH_10mM_AMUPOL_1p3mm_18kHz_DNP_150K"
)
dataset.expno_of_topspin_experiment = [24, 33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin-Glycin_150K_13mm"
dataset.peak_dict = {
    "171": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "48": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "32": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "25": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20231218_100mM_Ac_N-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
dataset.expno_of_topspin_experiment = [24, 33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Acetyl-Prolin_100K_13mm"
dataset.peak_dict = {
    "178": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "50": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "32": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "26": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20231218_100mM_Ac_N-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_125K"
)
dataset.expno_of_topspin_experiment = [24, 33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Acetyl-Prolin_125K_13mm"
dataset.peak_dict = {
    "178": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "50": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "32": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "26": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20231220_100mM_Ac_N-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
)
dataset.expno_of_topspin_experiment = [24, 25, 26, 27, 28, 29, 31, 32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Acetyl-Prolin_150K_13mm"
dataset.peak_dict = {
    "178": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "50": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "32": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "26": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20230817_100mM_HN-PA-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
dataset.expno_of_topspin_experiment = [24, 33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin-Alanin_100K_13mm"
dataset.peak_dict = {
    "171": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "48": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "32": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "26": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20230818_100mM_HN-PA-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
)
dataset.expno_of_topspin_experiment = [24, 33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin-Alanin_150K_13mm"
dataset.peak_dict = {
    "171": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "48": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "32": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "26": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20231026_100mM_HN-PA-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K_2"
)
dataset.expno_of_topspin_experiment = [24, 33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin-Alanin_150K_13mm_v2"
dataset.peak_dict = {
    "171": dict(sign="+", fitting_group=2, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "48": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "32": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "26": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -1
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = (
    r"F:\NMR\Max\20230815_1M_HN-P-OH_na_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
dataset.expno_of_topspin_experiment = [24, 32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Prolin_na_100K_13mm"
)
dataset.peak_dict = {
    #'175': dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "61": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "46": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "30": dict(sign="+", fitting_group=1, fitting_model="voigt"),
    "24": dict(sign="+", fitting_group=1, fitting_model="voigt"),
}
dataset._spectrum_number_for_prefit = -3
dataset.buildup_type = ["biexponential_with_offset", "biexponential"]
dataset.start_buildup_fit_from_topspin_export()
