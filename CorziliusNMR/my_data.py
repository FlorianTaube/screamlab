from CorziliusNMR import dataset as ds
import sys

list = [0,60,80,100,120,140,160,180,0,20,40,]


for i in list:
    print(i)
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20241021_S27F"
    dataset.expno_of_topspin_experiment =[i+1,i+8]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Scream_mod\S27F_{i+1}"
    dataset.peak_dict = {
        '-16': dict(sign="+", fitting_group=1, fitting_model="voigt"),
    }
    dataset._spectrum_number_for_prefit = -2
    dataset.buildup_type = ["biexponential_with_offset",
                         "exponential_with_offset"]
    dataset.start_buildup_fit_from_topspin_export()
sys.exit()
for i in list:
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20241024_S30F"
    dataset.expno_of_topspin_experiment =[i+1,i+7]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Scream_mod\S30F_{i+7}"
    dataset.peak_dict = {
        '19': dict(sign="+", fitting_group=1, fitting_model="voigt"),
    }
    dataset._spectrum_number_for_prefit = -3
    dataset.buildup_type = ["exponential_with_offset"]
    dataset.start_buildup_fit_from_topspin_export()























for i in list:
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20241022_S28F"
    dataset.expno_of_topspin_experiment =[i+1,i+7]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Scream_mod\S28F_{i+1}"
    dataset.peak_dict = {
        '-15': dict(sign="+", fitting_group=1, fitting_model="voigt"),
    }
    dataset._spectrum_number_for_prefit = -2
    dataset.buildup_type = ["biexponential_with_offset",
                         "exponential_with_offset"]
    dataset.start_buildup_fit_from_topspin_export()
for i in list:
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20241023_S29F"
    dataset.expno_of_topspin_experiment =[i+1,i+7]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Scream_mod\S29F_{i+1}"
    dataset._spectrum_number_for_prefit = -2
    dataset.peak_dict = {
        '-15': dict(sign="+", fitting_group=1, fitting_model="voigt"),
    }
    dataset.buildup_type = ["biexponential_with_offset",
                         "exponential_with_offset"]
    dataset.start_buildup_fit_from_topspin_export()



sys.exit()


dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241203_S36F_150K"
dataset.expno_of_topspin_experiment =[23,31]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_na_150K_3_2mm"
dataset.peak_dict = {
    '141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '27': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '11': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-5': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-8': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()
sys.exit()
dataset = ds.Dataset()
dataset.path_to_topspin_experiment = \
    r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K_eval_without_window"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_100K_1_3mm_eval_without_window_function"
dataset.peak_dict = {
    '173': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '60': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '44': dict(sign="+", fitting_group=4, fitting_mospdel="voigt"),
    '28': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '22': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()


sys.exit()



dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241118_S31F__100K_1"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_na_100K_3_2mm_verunreinigt"
dataset.peak_dict = {
    #'141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '28': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '12': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-4': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-9': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241119_S31F_150K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_na_150K_3_2mm_verunreinigt"
dataset.peak_dict = {
    '141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '27': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '11': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-4': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-9': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()


dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241118_S31F_125K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_na_125K_3_2mm_verunreinigt"
dataset.peak_dict = {
    '141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '27': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '11': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-4': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-8': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230816_1M_HN-P-OH_na_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
dataset.expno_of_topspin_experiment =[24,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_na_150K_1_3mm"
dataset.peak_dict = {
    '169': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '55': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '39': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '23': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '18': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()


dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241129_S36F_125K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_na_125K_3_2mm"
dataset.peak_dict = {
    '141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '27': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '11': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-6': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-9': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230815_1M_HN-P-OH_na_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
dataset.expno_of_topspin_experiment =[25,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_na_100K_1_3mm"
dataset.peak_dict = {
    #'173': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '54': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '40': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '24': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '18': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241128_S36F_100K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_na_100K_3_2mm"
dataset.peak_dict = {
    #'141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '27': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '11': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-6': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-9': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()



dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241126_S33F_125K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_125K_3_2mm"
dataset.peak_dict = {
    '141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '27': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '11': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-6': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-9': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()


dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241126_S33F_100K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_100K_3_2mm"
dataset.peak_dict = {
    '141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '27': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '11': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-6': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-9': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()


dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_100K_1_3mm"
dataset.peak_dict = {
    '173': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '60': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '44': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '28': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '22': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()



dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_125K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_125K_1_3mm"
dataset.peak_dict = {
    '174': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '60': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '46': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '29': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '24': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()


dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241127_S33F_150K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_150K_3_2mm"
dataset.peak_dict = {
    '141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '27': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '11': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-6': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-9': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()


dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_150K_1_3mm"
dataset.peak_dict = {
    '168': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '55': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '40': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '23': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '18': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()
