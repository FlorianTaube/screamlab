from CorziliusNMR import dataset as ds
import sys

list = [0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,8.0]
list=[8.0]
for i in list:
    print(i)
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = fr"F:\ssNMR\20250110_46F_{i}"
    dataset.expno_of_topspin_experiment =[1,18]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\S46F\S46F_{i}"
    dataset.peak_dict = {
        '-6': dict(sign="+", fitting_group=1, fitting_model="voigt"),
    }
    dataset._spectrum_number_for_prefit = -2
    dataset.buildup_type = ["biexponential_with_offset",
                         "exponential_with_offset","exponential"]
    dataset.start_buildup_fit_from_topspin_export()

for i in list:
    print(i)
    dataset = ds.Dataset()
    dataset.path_to_topspin_experiment = fr"F:\ssNMR\20250110_45F_{i}"
    dataset.expno_of_topspin_experiment =[1,18]
    dataset.procno_of_topspin_experiment = "103"
    dataset.output_file_name = fr"C:\Users\Florian Taube\Desktop\S45F\S45F_{i}"
    dataset.peak_dict = {
        '-6': dict(sign="+", fitting_group=1, fitting_model="voigt"),
    }
    dataset._spectrum_number_for_prefit = -2
    dataset.buildup_type = ["biexponential_with_offset",
                         "exponential_with_offset","exponential"]
    dataset.start_buildup_fit_from_topspin_export()