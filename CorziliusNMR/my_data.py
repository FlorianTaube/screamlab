from CorziliusNMR import dataset as ds
import sys

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\ssNMR\20241127_S35F_100K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Alanin_Prolin_100K"
dataset.peak_dict = {
    '145': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '27': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '13': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-5': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-9': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()
sys.exit()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
dataset.expno_of_topspin_experiment =[24,33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_100K"
dataset.peak_dict = {
    '173': dict(sign="-", fitting_group=2, fitting_model="voigt"),
    '58': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '44': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '28': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '22': dict(sign="-", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()
sys.exit()
dataset = ds.Dataset()
dataset.path_to_topspin_experiment = \
    r"F:\ssNMR\20241118_S31F_125K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_natural_abundance_125K_3_2mm_rotor"
dataset.peak_dict = {
    #'141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '27': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '13': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-3': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-8': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()












dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230816_1M_HN-P-OH_na_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
dataset.expno_of_topspin_experiment =[22,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_natural_abundance_150K"
dataset.peak_dict = {
    '169': dict(sign="-", fitting_group=2, fitting_model="voigt"),
    '54': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '40': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '23': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '17': dict(sign="-", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()










for i in range(10):
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20241021_S27F"
    dataset.expno_of_topspin_experiment = [1+i*20, 8+i*20]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Amupol_60_30_10_{1+i*20}_{8+i*20}"
    dataset.buildup_type = ["exponential_with_offset"]
    dataset.peak_dict = {
        '-16': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    }
    dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = \
    r"F:\ssNMR\20241119_S31F_150K"
dataset.expno_of_topspin_experiment =[23,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_natural_abundance_150K_3_2mm_rotor"
dataset.peak_dict = {
    #'141': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '27': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '13': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-3': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '-8': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()


dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230815_1M_HN-P-OH_na_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
dataset.expno_of_topspin_experiment =[22,33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_natural_abundance_100K"
dataset.peak_dict = {
    #'178': dict(sign="-", fitting_group=2, fitting_model="voigt"),
    '54': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '40': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '23': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '17': dict(sign="-", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()




dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230816_1M_HN-P-OH_na_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
dataset.expno_of_topspin_experiment =[22,32]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_natural_abundance_150K"
dataset.peak_dict = {
    '169': dict(sign="-", fitting_group=2, fitting_model="voigt"),
    '54': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '40': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '23': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '17': dict(sign="-", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20231218_100mM_Ac_N-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_125K"
dataset.expno_of_topspin_experiment =[22,33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Acetylprolin_125K"
dataset.peak_dict = {
    '178': dict(sign="-", fitting_group=2, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()















dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
dataset.expno_of_topspin_experiment =[22,33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_100K"
dataset.peak_dict = {
    '173': dict(sign="-", fitting_group=2, fitting_model="voigt"),
    '58': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '44': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '28': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '22': dict(sign="-", fitting_group=4, fitting_model="voigt"),
}

dataset.start_buildup_fit_from_topspin_export()
print("hello")
dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_125K"
dataset.expno_of_topspin_experiment =[22,33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_125K"
dataset.peak_dict = {
    '175': dict(sign="-", fitting_group=2, fitting_model="voigt"),
    '61': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '46': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '29': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '24': dict(sign="-", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()
print("hello")
dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
dataset.expno_of_topspin_experiment =[22,33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin_150K"
dataset.peak_dict = {
    '168': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    '54': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '40': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '24': dict(sign="+", fitting_group=4, fitting_model="voigt"),
    '18': dict(sign="+", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()
print("hello")






for i in range(10):
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20240607_S24F"
    dataset.expno_of_topspin_experiment = [2+i*20, 6+i*20]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Amupol_Ala_Pro_{1+i*20}_{11+i*20}"
    dataset.buildup_type = ["exponential_with_offset"]
    dataset.peak_dict = {
        '-3': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    }
    dataset.start_buildup_fit_from_topspin_export()


for i in range(10):
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20241021_S27F"
    dataset.expno_of_topspin_experiment = [1 + i * 20, 7 + i * 20]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Amupol_60_30_10_{1 + i * 20}_{7 + i * 20}"
    dataset.buildup_type = ["exponential_with_offset"]
    dataset.peak_dict = {
        '-16': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    }
    dataset.start_buildup_fit_from_topspin_export()

for i in range(10):
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20241022_S28F"
    dataset.expno_of_topspin_experiment = [1+i*20, 8+i*20]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Amupol_60_10_30_{1+i*20}_{8+i*20}"
    dataset.buildup_type = ["exponential_with_offset"]
    dataset.peak_dict = {
        '-16': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    }
    dataset.start_buildup_fit_from_topspin_export()

for i in range(10):
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20241022_S28F"
    dataset.expno_of_topspin_experiment = [1 + i * 20, 7 + i * 20]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Amupol_60_10_30_{1 + i * 20}_{7 + i * 20}"
    dataset.buildup_type = ["exponential_with_offset"]
    dataset.peak_dict = {
        '-16': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    }
    dataset.start_buildup_fit_from_topspin_export()


for i in range(10):
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20241023_S29F"
    dataset.expno_of_topspin_experiment = [1+i*20, 8+i*20]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Asympol_60_30_10{1+i*20}_{8+i*20}"
    dataset.buildup_type = ["exponential_with_offset"]
    dataset.peak_dict = {
        '-16': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    }
    dataset.start_buildup_fit_from_topspin_export()

for i in range(10):
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20241023_S29F"
    dataset.expno_of_topspin_experiment = [1 + i * 20, 7 + i * 20]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Asympol_60_30_10{1 + i * 20}_{7 + i * 20}"
    dataset.buildup_type = ["exponential_with_offset"]
    dataset.peak_dict = {
        '-16': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    }
    dataset.start_buildup_fit_from_topspin_export()

for i in range(10):
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20241024_S30F"
    dataset.expno_of_topspin_experiment = [1+i*20, 8+i*20]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Asympol_60_10_30{1+i*20}_{8+i*20}"
    dataset.buildup_type = ["exponential_with_offset"]
    dataset.peak_dict = {
        '-16': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    }
    dataset.start_buildup_fit_from_topspin_export()

for i in range(10):
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = r"F:\ssNMR\20241024_S30F"
    dataset.expno_of_topspin_experiment = [1 + i * 20, 7 + i * 20]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Asympol_60_10_30{1 + i * 20}_{7 + i * 20}"
    dataset.buildup_type = ["exponential_with_offset"]
    dataset.peak_dict = {
        '-16': dict(sign="+", fitting_group=2, fitting_model="voigt"),
    }
    dataset.start_buildup_fit_from_topspin_export()














dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230620_100mM_H2N-PG-OH_10mM_AMUPOL_1p3mm_18kHz_DNP_100K"
dataset.expno_of_topspin_experiment =[22,33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin-Glycin_100K"
dataset.peak_dict = {
    '172': dict(sign="-", fitting_group=2, fitting_model="voigt"),
    '61': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '48': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '32': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '26': dict(sign="-", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()

dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230814_100mM_H2N-PG-OH_10mM_AMUPOL_1p3mm_18kHz_DNP_150K"
dataset.expno_of_topspin_experiment =[22,33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin-Glycin_150K"
dataset.peak_dict = {
    '172': dict(sign="-", fitting_group=2, fitting_model="voigt"),
    '61': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '48': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '32': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '26': dict(sign="-", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()
print("hello")
dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230817_100mM_HN-PA-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
dataset.expno_of_topspin_experiment =[22,33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin-Alanin_100K"
dataset.peak_dict = {
    '163': dict(sign="-", fitting_group=2, fitting_model="voigt"),
    '53': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '40': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '24': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '18': dict(sign="-", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()
print("hello")
dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20230818_100mM_HN-PA-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
dataset.expno_of_topspin_experiment =[22,33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Prolin-Alanin_150K"
dataset.peak_dict = {
    '163': dict(sign="-", fitting_group=2, fitting_model="voigt"),
    '53': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '40': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '24': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '18': dict(sign="-", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()
print("hello")
dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20231218_100mM_Ac_N-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
dataset.expno_of_topspin_experiment =[22,33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Acetylprolin_100K"
dataset.peak_dict = {
    '178': dict(sign="-", fitting_group=2, fitting_model="voigt"),
    '61': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '50': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '32': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '26': dict(sign="-", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()


print("hello")

print("hello")
dataset = ds.Dataset()
dataset.path_to_topspin_experiment = r"F:\NMR\Max\20231220_100mM_Ac_N-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
dataset.expno_of_topspin_experiment =[22,23,24,25,26,27,28,29,31,32,33]
dataset.procno_of_topspin_experiment = "103"
dataset.output_file_name = r"C:\Users\Florian " \
                           r"Taube\Desktop\Prolin_auswertung\Acetylprolin_150K"
dataset.peak_dict = {
    '178': dict(sign="-", fitting_group=2, fitting_model="voigt"),
    '61': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '50': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '32': dict(sign="-", fitting_group=4, fitting_model="voigt"),
    '26': dict(sign="-", fitting_group=4, fitting_model="voigt"),
}
dataset.start_buildup_fit_from_topspin_export()

print("hello")


