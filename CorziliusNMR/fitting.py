"""
fitting module of the CorziliusNMR package.
"""
# Importing necessary packages
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
from lmfit.models import ExpressionModel, VoigtModel
from scipy import special
from scipy.special import wofz
from lmfit import Parameters, minimize, fit_report
from pyDOE import lhs
import sys
import os
import csv
import lmfit

def generate_csv(output_file):
    return output_file+".csv"

def check_csv_name_for_backslash(csv_name):
    return csv_name.replace("\\", "/") if "\\" in csv_name else csv_name

def get_delay_times_from_csv(csv_name):
    with open(csv_name, 'r') as file:
        for count, row in enumerate(csv.reader(file)):
            if count == 10:
                return [float(value.strip(' #[].')) for value in row if float(value.strip(' #[].')) != 0]

def read_xy_data_from_csv(csv_file_name):
    data = np.loadtxt(csv_file_name, delimiter=",")
    # Sort input y_data into x_data and y values -> even columns = x_data-axis, uneven columns= y-axis
    x_data = [data[:, i] for i in range(0, np.size(data, 1)) if i % 2 == 0]
    y_data = [data[:, i] for i in range(0, np.size(data, 1)) if i % 2 != 0]
    return x_data,y_data

def check_if_autopeakpick_is_possible(plus_minus):
    return all(element == plus_minus[0] for element in plus_minus)

def horst(peak_list,plus_minus_list):
    return  peak_list,plus_minus_list

def auto_peak_pick(peak_list,plus_minus_list,x_data,y_data):
    height = 1
    multiplier = int(plus_minus_list[0] + "1")
    peaks, _ = find_peaks(multiplier * y_data[-1], height=height,
                          distance=150)
    # Increase height until the number of detected peaks is sufficient
    while len(peaks) > len(peak_list):
        height += 1
        peaks, _ = find_peaks(multiplier * y_data[-1], height=height, distance=150)
    return [int(x_data[-1][peak]) for peak in peaks]

def auto_peak_pick_value(peak_list,plus_minus_list,x_data,y_data):
    height = 1
    multiplier = int(plus_minus_list[0] + "1")
    peaks, _ = find_peaks(multiplier * y_data, height=height,
                          distance=150)
    # Increase height until the number of detected peaks is sufficient
    while len(peaks) > len(peak_list):
        height += 1
        peaks, _ = find_peaks(multiplier * y_data, height=height, distance=150)
    return [float(y_data[peak]) for peak in peaks]
def generate_peak_label(peak_list):
    carbon_groups = []
    integral = dict()
    for peak in peak_list:
        mod_peak = str(peak)
        if "-" in mod_peak:
            mod_peak = mod_peak.replace("-","m")
        integral[f"Peak_at_{mod_peak}_ppm"] = []
        carbon_groups.append(f"Peak_at_{mod_peak}_ppm")
    return carbon_groups,integral

def calculate_max_for_each_peak(x_data,y_data,peak_labels,peak_list,integral,plus_minus_list):
    peak_value = []
    for nr, spectrum in enumerate(y_data):
        peak_value.append(auto_peak_pick_value(peak_list, plus_minus_list, x_data[nr], y_data[nr]))

    for val_list_nr, value_list in enumerate(peak_value):
        for val_nr, value in enumerate(value_list):
            integral[peak_labels[val_nr]].append(value)
    return integral

def fit_lhs(peak_center,plus_minus,x_data,y_data,sample,minmax):
    voigtans = []
    model = ''
    for peak_nr, peak in enumerate(peak_center):
        peak_str = str(peak).replace('-', 'm') if '-' in str(peak) else peak
        voigtans.append(VoigtModel(prefix=f'Peak_at_{peak_str}_ppm'))
        amp_value = -200 if '-' in plus_minus[peak_nr] else 200
        param_dict = dict(
            center=dict(value=peak, min=peak - 2, max=peak + 2, vary=True),
            amplitude=dict(value=amp_value, max=0 if amp_value < 0 else None,
                           min=0 if amp_value > 0 else None),
            sigma=dict(value=sample[peak_nr], min=minmax[0],
                       max=minmax[1],vary=True),
            gamma=dict(value=sample[peak_nr+1], min=minmax[0],
                       max=minmax[1],vary=True)
        )

        if peak_nr == 0:
            params = voigtans[peak_nr].make_params(**param_dict)
            model = voigtans[peak_nr]
        else:
            params.update(voigtans[peak_nr].make_params(**param_dict))
            model += voigtans[peak_nr]
    return model.fit(y_data, params, x=x_data,nan_policy='omit')
def voigt_fit_with_lhs(peak_center,plus_minus,x_data,y_data,minmax=(0.0,3)):
    n_samples = 1*1#*#len(peak_center)
    param_ranges = dict()

    for peak in peak_center:
        param_ranges["sigma_"+str(peak)] = minmax
        param_ranges["gamma_" + str(peak)] = minmax

    # Generate LHS samples in normalized space [0, 1]
    lhs_samples = lhs(len(param_ranges), samples=n_samples)

    # Scale LHS samples to the actual parameter ranges
    param_names = list(param_ranges.keys())
    param_samples = np.zeros_like(lhs_samples)
    for i, param_name in enumerate(param_names):
        lower, upper = param_ranges[param_name]
        param_samples[:, i] = lhs_samples[:, i] * (upper - lower) + lower

    best_fit = None
    best_residual = np.inf

    for sample_nr,sample in enumerate(param_samples):
        print(sample_nr)
        result = fit_lhs(peak_center,plus_minus,x_data,y_data,sample,minmax)
        residual = np.sum(result.residual ** 2)
        if residual < best_residual:
            best_residual = residual
            best_result = result
    return best_result

def voigt(x, amp, cen, sig, gam):
    z = (x - cen + 1j * gam) / (sig * np.sqrt(2))
    return amp * special.wofz(z).real / (sig * np.sqrt(2 * np.pi))

def voigt_old(x, amp, cen, sig, gam):
    return sum(
        amp[nr] * special.wofz((x-cen[nr]+1j*gam[nr])/(sig[nr]*np.sqrt(2))).real / (sig[nr] * np.sqrt(2 * np.pi))
        for nr in range(len(amp))
    )


def calc_integrals_from_fitting_results(peak_labels,best_results,x_data):
    integral = dict()
    for peak in peak_labels:
        integral[peak] = []
    for result_nr,result in enumerate(best_results):
        for peak_nr, peak in enumerate(peak_labels):
            amp, cen, sig, gam = (best_results[result_nr].get(peak + key) for key in
                                  ["amplitude", "center", "sigma", "gamma"])
            sim_y_data = voigt(x_data[result_nr],amp,cen,sig,gam)
            plt.plot(x_data[result_nr],sim_y_data,linestyle='-')
            integral[peak].append(np.trapz(sim_y_data))
    plt.close()
    return integral


def calculate_voigt_for_each_peak(x_data,y_data,peak_labels,peak_list,integral,plus_minus_list):
    best_results = dict()
    for spectrum_nr, spectrum in enumerate(y_data):
        best_result = voigt_fit_with_lhs(peak_list,plus_minus_list,x_data[spectrum_nr],y_data[spectrum_nr])
        plt.plot(x_data[spectrum_nr],y_data[spectrum_nr], color="#000000")
        plt.plot(x_data[spectrum_nr], best_result.best_fit, color='#FF0000', linestyle='--')
        best_results[spectrum_nr] = best_result.best_values
    #plt.show()
    plt.close()
    return calc_integrals_from_fitting_results(peak_labels,best_results,x_data)

def get_model_exp(fitting_type):
    if fitting_type == "Exponential":
        return 'off + A1*(1-exp(-(x)/x1))'
    elif fitting_type == "Biexponential:":
        return 'off + A_fast*(1-exp(-(x)/x_fast)) + A_slow*(1-exp(-(x)/x_slow))'
    elif fitting_type == "Exponential_with_offset":
        return  'off + A_fast*(1-exp(-(x-t0)/x_fast))'
    elif fitting_type == "Biexponential_with_offset":
        return 'off + A_fast*(1-exp(-(x-t0)/x_fast)) + A_slow*(1-exp(-(x-t0)/x_slow))'
    elif fitting_type == "Exponential_Test":
        return "A2 * (exp((x + t1) / x2) - exp(t1 / x2))"
    elif fitting_type == 'Sig_weight':
        return "1 / (1 + np.exp(-k * (x - x_transition))) * A2 * (np.exp((x + t1) / x2) - np.exp(t1 / x2)) + (1 - 1 / (1 + np.exp(-k * (x - x_transition)))) * A_fast * (1 - np.exp(-(x_data - t0) / x_fast))"
    elif fitting_type == 'Solomon':
        return "M_C_inf * (1 - exp(-(x)/lambda_H )) * (1 - exp(-(x)/lambda_C))"
    elif fitting_type == 'Solomon_biexp':
        return "M_C_inf * (1 - exp(-(x)/lambda_H ))* (A1*(1 - exp(-(x)/lambda_C))+ (1-A1)*(1-exp(-(x)/lambda_C2)))"

def get_param_dict(fitting_type):
    if fitting_type == "Exponential":
        return {'off': dict(value=0, vary=False),
        'x1': dict(value=15),
        'A1': 500000 }
    elif fitting_type == "Biexponential":
        return {'off': dict(value=0, vary=False),
        'x_fast': dict(value=15, min=1, max=200),
        'x_slow': dict(value=200, min=0, max=1000),
        'A_fast': 500000,
           'A_slow': 1000000 }
    elif fitting_type == "Exponential_with_offset":
        return {'off': dict(value=0, vary=False),
        't0': dict(value=0, min=-4, max=4),
        'x_fast': dict(value=15),
        'A_fast': 500000}
    elif fitting_type == "Biexponential_with_offset":
        return {'off': dict(value=0, vary=False),
        't0': dict(value=0, min=-4, max=4),
        'x_fast': dict(value=15, min=1, max=200),
        'x_slow': dict(value=200, min=0, max=1000),
        'A_fast': 500000,
        'A_slow': 1000000}
    elif fitting_type == "Exponential_Test":
        return {'t1': dict(value=-3.42080032, vary=False),
        'x2': dict(value=1.86274121 , vary=False),
        'A2': dict(value=1135.16566, vary=False) ,}
    elif fitting_type == "Solomon":
        return { "M_C_inf":5000,
                "lambda_H":dict(value=3, min=0),
                "lambda_C":dict(value=20,min=0),}
    elif fitting_type == 'Solomon_biexp':
        return { "M_C_inf":dict(value=5000),
                "lambda_H":dict(value=3, min=0,  max=50),
                "lambda_C":dict(value=20,min=0,max=100),
                "lambda_C2": dict(value=200, min=0,max=1000),
                 "A1":dict(value=0.5,min=0,max=1),}

def lhs_buildup_fit(integral,buildup_fit_model, param_dict,delay_times):
    n_samples = 200
    param_ranges = dict()
    param_ranges["M_C_inf"] = (0, max(delay_times)*2)
    param_ranges["lambda_H"] = (0,50)
    param_ranges["lambda_C"] = (0, 100)
    param_ranges["lambda_C2"] = (0, 1000)
    param_ranges["A1"]=(0,1)

    lhs_samples = lhs(len(param_ranges), samples=n_samples)
    param_names = list(param_ranges.keys())
    param_samples = np.zeros_like(lhs_samples)
    for i, param_name in enumerate(param_names):
        lower, upper = param_ranges[param_name]
        param_samples[:, i] = lhs_samples[:, i] * (upper - lower) + lower
    best_fit = None
    best_residual = np.inf
    for sample_nr, sample in enumerate(param_samples):
        for key_nr,key in enumerate(list(param_dict.keys())):
            param_dict[key]["value"] = sample[key_nr]
        params = buildup_fit_model.make_params(**param_dict)
        result = buildup_fit_model.fit(integral, params, x=delay_times)
        residual = np.sum(result.residual ** 2)
        if residual < best_residual:
            best_residual = residual
            best_result = result
    return best_result

def exp_or_biexp_fit(delay_times,peak_list,peak_lables,integral,fitting_type,plot_errorbar=False,lhs=True):
    cmap = plt.cm.viridis
    for peak_nr,peak in enumerate(peak_lables):
        integral[peak] = [abs(val) for val in integral[peak]]
        model_expr = get_model_exp(fitting_type)
        buildup_fit_model = ExpressionModel(model_expr)
        param_dict = get_param_dict(fitting_type)
        params = buildup_fit_model.make_params(**param_dict)
        if lhs:
            result = lhs_buildup_fit(integral[peak],buildup_fit_model, param_dict,delay_times)
        else:
            result = buildup_fit_model.fit(integral[peak], params, x=delay_times)
        print(result.fit_report())
        color = cmap(peak_nr / len(peak_lables))
        plt.plot(delay_times, integral[peak], 'o', color=color)
        tau = np.arange(0, delay_times[-1] +2, 0.01)
        dely = result.eval_uncertainty(sigma=3)
        plt.plot(tau, result.eval(x=tau), label=str(peak), color=color)
        if plot_errorbar:
            plt.errorbar(delay_times, integral[peak], yerr=dely, fmt='o', color=color,  capsize=3)
    plt.legend()
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.show()


def get_prefit_params(peak_center, plus_minus,x_data,y_data,peak_labels):
    return voigt_fit_with_lhs(peak_center, plus_minus, x_data, y_data)


def generate_parameters(peak_labels, y_data, prefit_params, plus_minus, global_fit, peak_list):
    params = Parameters()
    for peak_nr, peak in enumerate(peak_labels):
        for time_nr, y in enumerate(y_data):
            fitting_key = peak
            old_fitting_key =f'Peak_at_{peak_list[peak_nr]}_ppm'
            if "-" in plus_minus[peak_nr]:
                params.add(f'{fitting_key}_amp_{time_nr + 1}', value=-200, max=0, vary=True)
            else:
                params.add(f'{fitting_key}_amp_{time_nr + 1}', value=200, min=0, vary=True)
            params.add(f'{fitting_key}_cen_{time_nr + 1}',
                       value=prefit_params[old_fitting_key + "center"] - 0.01,
                       min=prefit_params[old_fitting_key + "center"] - 0.2,
                       max=prefit_params[old_fitting_key + "center"] + 0.2, vary=True)
            params.add(f'{fitting_key}_sig_{time_nr + 1}', value=prefit_params[old_fitting_key + "sigma"],
                       min=prefit_params[old_fitting_key + "sigma"] - 0.1 * prefit_params[old_fitting_key + "sigma"],
                       max=prefit_params[old_fitting_key + "sigma"] + 0.1 * prefit_params[old_fitting_key + "sigma"],
                       vary=True)
            params.add(f'{fitting_key}_gam_{time_nr + 1}', value=prefit_params[old_fitting_key + "gamma"],
                       min=prefit_params[old_fitting_key + "gamma"] - 0.1 * prefit_params[old_fitting_key + "gamma"],
                       max=prefit_params[old_fitting_key + "gamma"] + 0.1 * prefit_params[old_fitting_key + "gamma"],
                       vary=True)
        # Set expressions for shared parameters
        if global_fit:
            for time_nr in range(1, len(y_data)):
                for param in ['sig', 'gam']:
                    params[f'{fitting_key}_{param}_{time_nr + 1}'].expr = f'{fitting_key}_{param}_1'
    return params


def voigt_dataset(fit_params, i, x_data, peak_lables):
    amp, cen, gam, sig = [], [], [], []
    for peak in peak_lables:
        for attr, lst in zip(['amp', 'cen', 'gam', 'sig'], [amp, cen, gam, sig]):
            lst.append(fit_params[f'{peak}_{attr}_{i + 1}'])
    return voigt_old(x_data, amp, cen, sig, gam)  # Calculate the Voigt profile for given parameters

# Objective function for Voigt fit
def voigt_objective(fit_params, x_data, y_data, peak_labels):
    resid = []
    for spectrum_nr, spectrum in enumerate(y_data):
        resid.append(y_data[spectrum_nr] - voigt_dataset(fit_params, spectrum_nr, x_data[spectrum_nr], peak_labels))
        #plt.plot(x_data[spectrum_nr],resid[spectrum_nr])
    #plt.show()
    #plt.close()
    return np.asarray(resid).flatten()

def calculate_global_voigt(peak_list, plus_minus_list,x_data,y_data,peak_labels,prefit_params,list_of_list,global_fit=True):
    integral = dict()
    for label in peak_labels:
        integral[label] = []
    for element_nr,element in enumerate(list_of_list):
        if type(element) == int:
            fit_params = generate_parameters([peak_labels[element_nr]], y_data, prefit_params, plus_minus_list, global_fit,peak_list)
            result = minimize(voigt_objective, fit_params, args=(x_data, y_data, [peak_labels[element_nr]]))
            for i in range(len(x_data)):
                y_fit = voigt_dataset(result.params, i, x_data[i], [peak_labels[element_nr]])
                integral[peak_labels[element_nr]].append(np.trapz(y_fit))

                plt.plot(x_data[i], y_data[i], color='#000000', label='Experiments')
                plt.plot(x_data[0], y_fit, linestyle="--", color='#FF0000', label='Simulations')
            plt.close()
        elif type(element) == list:
            selected_peak_labels = []
            selected_peak_list = []
            for value in element:
                selected_peak_labels.append(peak_labels[value-1])
                selected_peak_list.append(peak_list[value-1])
            fit_params = generate_parameters(selected_peak_labels, y_data, prefit_params, plus_minus_list,
                                             global_fit, selected_peak_list)
            result = minimize(voigt_objective, fit_params,
                              args=(np.asarray(x_data), np.asarray(y_data), selected_peak_labels))
            for i in range(len(x_data)):
                y_fit_calc  = voigt_dataset(result.params, i, x_data[i], selected_peak_labels)
                plt.plot(x_data[i], y_data[i], color='#000000', label='Experiments')
                plt.plot(x_data[i], y_fit_calc, linestyle="--", color='#FF0000', label='Simulations')

                for peak_nr,peak_label in enumerate(selected_peak_labels):
                    calc_param = Parameters()
                    for variable in ['amp','cen','sig','gam']:
                        calc_param.add(f'{peak_label}_{variable}_{i+1}', value=result.params[f'{peak_label}_{variable}_{i+1}'].value)
                    integral[peak_label].append(np.trapz(voigt_dataset(calc_param,i,x_data[i],[peak_label])))
            #plt.show()
            plt.close()
            return integral

def main(output_file,peak_list, plus_minus_list=["+"],evaluation_type="voigt",autopeakpick=True,fitting_type = ["Solomon"],list_of_list = [1,[2,3,4,5]]):
    csv_file_name = generate_csv(output_file)
    csv_file_name = check_csv_name_for_backslash(csv_file_name)
    delay_times = get_delay_times_from_csv(csv_file_name)
    x_data, y_data = read_xy_data_from_csv(csv_file_name)
    peak_list, plus_minus_list = horst(peak_list,plus_minus_list)
    if autopeakpick:
        if check_if_autopeakpick_is_possible(plus_minus_list):
            peak_list = auto_peak_pick(peak_list,plus_minus_list,x_data,y_data)
        else:
            print("ERROR: Auto peak picking is not possible if some peaks are positive and some are negative.")
    peak_labels, integral = generate_peak_label(peak_list)
    if evaluation_type == "max_value":
        integral = calculate_max_for_each_peak(x_data,y_data,peak_labels,peak_list,integral,plus_minus_list)
    elif evaluation_type == "voigt":
        integral = calculate_voigt_for_each_peak(x_data,y_data,peak_labels,peak_list,integral,plus_minus_list)
    elif evaluation_type == "global_voigt":
        prefit_params = get_prefit_params(peak_list, plus_minus_list,x_data[-3],y_data[-3],peak_labels).values
        integral = calculate_global_voigt(peak_list, plus_minus_list,x_data,y_data,peak_labels,prefit_params,list_of_list)
    for type in fitting_type:
        exp_or_biexp_fit(delay_times,peak_list,peak_labels,integral,type)











































# Calculate FWHM from fitted linewidth fit_params
def calc_fwhm(params):
    FWHM_Lorentzian = 2 * params[3]
    FWHM_Gaussian = 2.35482 * params[2]
    FWHM_Voigt = 0.5346 * FWHM_Lorentzian + np.sqrt(0.2166 * FWHM_Lorentzian**2 + FWHM_Gaussian**2)
    return FWHM_Voigt

# Voigt function definition


# Voigt dataset calculation

# Calculation of simulated spectrum


#Funktion funktioniert noch nicht
def print_tabel_voigt(fit_report, delay_times, outtxt):
    fit_lines = fit_report.split("\n")
    extracting = False
    variables_section = []
    printout = []
    printout_error = []
    # Extract variables section
    for line in fit_lines:
        if '[[Variables]]' in line:
            extracting = True
            continue
        if '[[Correlations]]' in line:
            extracting = False
            break
        if extracting:
            parts = line.strip().split("(")[0].split("_")[3:]
            variables_section.append(parts)

    printout.append("Name\tt_pol / s\tAmplitude / a.u.\tdelta / ppm\tsigma / ppm\tgamma / ppm\tFWHM / ppm")

    # Process variables
    count = 1
    for i in range(0, len(variables_section), 4):
        fwhm_list = []
        fwhm_error = []
        for j in range(i, i + 4):
            if j < len(variables_section):
                var = variables_section[j]
                fwhm_list.append(float(var[-1].split("+/-")[0].split(":")[-1]))
                fwhm_error.append(var[-1].split(":")[-1])

        if fwhm_list:
            fwhm_list.append(calc_fwhm(fwhm_list))
            delay_time = delay_times[int(variables_section[i][2].split(":")[0]) - 1]
            printout.append(f"{variables_section[i][0]}\t{delay_time}\t" + "\t".join(map(str, fwhm_list)))
            printout_error.append(f"{variables_section[i][0]}\t{delay_time}\t" + "\t".join(map(str, fwhm_error)))
            count += 1

    # Write output to file
    with open(outtxt, 'w') as file:
        file.write("\n".join(printout) + "\n\n")
        file.write("\n".join(printout_error) + "\n")

# Output of a tabel with all buildup parameters
def get_biexp_fit_pars(fit_report):
    fit_lines = fit_report.split("\n")
    extracting = False
    variables_section = []
    # Extract variables section
    for line in fit_lines:
        if '[[Variables]]' in line:
            extracting = True
            continue
        if '[[Correlations]]' in line:
            extracting = False
            break
        if extracting:
            parts = line.strip().split("(")[0].split(":")[-1]
            variables_section.append(parts)
    variables_section.pop(0)
    variables_section = [float(element.split("+/-")[0]) for element in variables_section]
    if len(variables_section) == 4:
        variables_section.extend([
            variables_section[0] / math.sqrt(variables_section[1]),
            variables_section[2] / math.sqrt(variables_section[3])
        ])
    else:
        variables_section.append(variables_section[0] / math.sqrt(variables_section[1]))
    
    # Round all elements in variables_section to 2 decimal places
    variables_section = [round(element, 2) for element in variables_section]                 
    return "\t".join(map(str, variables_section))



def start_voigt_fitting(output_file,peak_center,peaks_for_single_fit=[],
                        peaks_for_multiple_fit=[],plus_minus=["+"],
                        autopeakpick=False,global_fit=False):
    input_file = output_file +".csv"
    plus_minus = plus_minus*len(peak_center)

    if '\\' in input_file:
        input_file = input_file.replace("\\","/")
    count = 0
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if count == 10:  # Stop after 10 rows
                delay_times = [float(value.strip(' #[].')) for value in row]
                break
            count += 1
    delay_times = [x for x in delay_times if x != 0]
    # Reading input y_data
    data = np.loadtxt(input_file, delimiter=",")
    
    
    # Extracting dipeptide sequence from csv filename
    # Initialize lists for x_data- and y- y_data of NMR spectra and dictionary for integral of all peaks
    x_data = []
    y_data = []
    
    # Sort input y_data into x_data and y values -> even columns = x_data-axis, uneven columns= y-axis
    x_data = [data[:, i] for i in range(0, np.size(data, 1)) if i % 2 == 0]
    y_data = [data[:, i] for i in range(0, np.size(data, 1)) if i % 2 != 0]   
    if autopeakpick:
        height = 10
        multiplier = int(plus_minus[0]+"1")
        
        peaks, _ = find_peaks(multiplier*y_data[-1], height=height,
                              distance=150)
        # Increase height until the number of detected peaks is sufficient
        while len(peaks) > len(peak_center):
            height += 1
            peaks, _ = find_peaks(multiplier*y_data[-1], height=height, distance=150)
        
        # Get the corresponding peak centers from x_data
        peak_center = [int(x_data[-1][peak]) for peak in peaks]
    carbon_groups = []
    integral = dict([])
    for peak in peak_center:
        mod_peak = str(peak)
        if "-" in mod_peak:
            mod_peak = mod_peak.replace("-","m")
        integral[f"Peak_at_{mod_peak}_ppm"] = []
        carbon_groups.append(f"Peak_at_{mod_peak}_ppm")
    # Convert y_data to numpy array
    x_data_array = np.asarray(x_data)
    y_data_array = np.asarray(y_data)
    spectrum_for_prefit = -1
    result = first_fit_with_lhs(peak_center,plus_minus,x_data_array,
                                y_data_array,spectrum_for_prefit)
    fitting_parameters = result.values
    for peak in carbon_groups:
        while fitting_parameters[peak+"sigma"] < 0.01:
            spectrum_for_prefit = spectrum_for_prefit -1
            result = first_fit_with_lhs(peak_center, plus_minus, x_data_array,
                                        y_data_array, spectrum_for_prefit)
            fitting_parameters = result.values
    fit = result.eval_components(x=x_data_array[-1])
    bestfit = 0
    for peak_nr,peak in enumerate(peak_center):
        if "-" in str(peak):
            peak_center[peak_nr] = str(peak_center[peak_nr]).replace('-','m')
    for peak_nr,peak in enumerate(peak_center):
        bestfit += fit[f'Peak_at_{peak}_ppm']
      
    plt.plot(x_data_array[-1], y_data_array[-1],color='#000000',
             label='Experiment')
    plt.plot(x_data_array[spectrum_for_prefit], bestfit,color='#FF0000', label='Simulation',
             linestyle='--')
    #plt.xlim([-20,-10])
    plt.legend(loc='lower center', fontsize=12)
    plt.xlabel('$chemical\ shift$ / ppm')
    plt.ylabel('$NMR\ signal\ intensity$ / a.u.')
    outfile = output_file+ "_first_fit.pdf"
    plt.savefig(outfile, dpi='figure', format="pdf")
    plt.close()

    single_fit_peaks = []
    single_plus_minus = []
    for peak in peaks_for_single_fit:
        single_fit_peaks.append(peak_center[peak-1])
        single_plus_minus.append(plus_minus[peak-1])
    # Loop over peak centers
    for number, peak in enumerate(single_fit_peaks):
        single_fit_peak = []
        single_fit_peak.append(peak)
        outtxt = output_file + "_deconvolution_fit_for_peak_at_" + str(peak) \
                 + "_ppm.txt"
    
        # Initialize fit parameters for individual fitting
        fit_params = generate_parameters(single_fit_peak, y_data,delay_times, fitting_parameters,single_plus_minus,global_fit)
    
        # Perform individual fit

        out = minimize(voigt_objective, fit_params, args=(x_data_array, y_data_array,single_fit_peak))
        # Write fit report to output file
        with open(outtxt, 'w') as file:
            file.write(fit_report(out))
        outtxt = output_file + "_deconvolution_fit_for_peak_at_" + str(peak) \
                 + "_ppm_tab.txt"
       # print_tabel_voigt(fit_report(out),delay_times,outtxt)
    
        # Plot the fit and save to output file
        plt.figure()

        for i in range(len(x_data)):
            outfile =  output_file +"_deconvolution_fit_for_peak_at_" + str(
                peak) + \
                       "_ppm.pdf"
            y_fit = voigt_dataset(out.params, i, x_data_array[i],
                                  single_fit_peak)
            y_data_array[i] = y_data_array[i] - y_fit
            integral[f"Peak_at_{peak}_ppm"].append(np.trapz(y_fit))
            plt.plot(x_data[i], y_data[i],color='#000000',label='Experiments')
            plt.plot(x_data[i], y_fit, linestyle="--",color='#FF0000',
                     label='Simulations')
            xlim_center = single_fit_peaks[number]

            if "m" in str(xlim_center):
                xlim_center = int(xlim_center.replace("m","-"))
            plt.xlim(xlim_center+50, xlim_center-50)
            #plt.xlim([200,0])
        plt.xlabel('$chemical\ shift$ / ppm')
        plt.ylabel('$NMR\ signal\ intensity$ \ a.u.')
        plt.savefig(outfile, dpi='figure', format="pdf")
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:2], labels[:2], fontsize=12)
        plt.close()

    # Print the carbon groups being processed
    if peaks_for_multiple_fit != []:
        multiple_fit_peaks = []
        multiple_plus_minus = []
        peak_str = ""
        for peak in peaks_for_multiple_fit:
            peak_str = peak_str + str(peak_center[peak-1])+"_"
            multiple_fit_peaks.append(peak_center[peak-1])
            multiple_plus_minus.append(plus_minus[peak-1])
        # Define output filenames
        outfile = output_file + "_fit_for_peaks_at_"+ peak_str +"_ppm.pdf"
        outtxt = output_file + "_deconvolution_fit_for_peaks_at_"+peak_str+"_ppm" \
                                                                           ".txt"

        # Initialize fit parameters for individual fitting
        fit_params = generate_parameters(multiple_fit_peaks, y_data, delay_times, fitting_parameters,multiple_plus_minus,global_fit)

        # Perform individual fit
        out = minimize(voigt_objective, fit_params, args=(x_data[0], y_data_array,multiple_fit_peaks))

        # Write fit report to output file
        with open(outtxt, 'w') as file:
            file.write(fit_report(out))
        outtxt =  output_file + "_deconvolution_fit_for_peak_at_" + peak_str + \
                  "_ppm_tab.txt"

        #print_tabel_voigt(fit_report(out),delay_times,outtxt)
        # Plot the fit and save to output file
        plt.figure()
        for i in range(len(x_data)):
            y_fit_calc = []
            y_fit_calc_sum = 0
            for nr, peak in enumerate(multiple_fit_peaks):
                y_fit_calc.append(calc_sim_spectrum(out, peak,x_data[0],i))
                y_fit_calc_sum = y_fit_calc_sum + y_fit_calc[nr]

            plt.plot(x_data[0], y_data[i],color='#000000',label='Experiments')
            plt.plot(x_data[0], y_fit_calc_sum, linestyle="--",color='#FF0000',label='Simulations')
            xlim_center1 = multiple_fit_peaks[0]
            xlim_center2 = multiple_fit_peaks[-1]
            if "m" in str(xlim_center1):
                xlim_center1 = int(xlim_center1.replace("m","-"))
            if "m" in str(xlim_center2):
                xlim_center2 = int(xlim_center2.replace("m","-"))
            plt.xlim((xlim_center2+xlim_center1)/2+40, (xlim_center2+xlim_center1)/2-40)

            for nr,peak in enumerate(multiple_fit_peaks):
                integral[f"Peak_at_{peak}_ppm"].append(np.trapz(y_fit_calc[nr]) * -1)


        plt.xlabel('$chemical\ shift$ / ppm')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:2], labels[:2], fontsize=12)
        plt.ylabel('$NMR\ signal\ intensity$ / a.u.')
        # Save the plot to the output file

        plt.savefig(outfile, dpi='figure', format="pdf")
        plt.close()
    
     
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=["#142C5F", "#124E63", "#2E6A57", "#F6A895", "#FFBAD7"])
    cmap = plt.cm.viridis   
    
    # Helper function to plot and fit
    def plot_and_fit(carbon_groups, delay_times, integral, output_file, model_expr, param_dict, fit_type):
        fit_par_tab = dict()
        for i, group in enumerate(carbon_groups):
            # Adjust integral values
            integral[group] = [abs(val) for val in integral[group]]
            color = cmap(i / len(carbon_groups))
            
            # Plot y_data points
            plt.plot(delay_times, integral[group], 'o', color=color)
            
            # Build the fit model
            buildup_fit_model = ExpressionModel(model_expr)
            params = buildup_fit_model.make_params(**param_dict)
            
            # Perform the fit
            result = buildup_fit_model.fit(np.array(integral[group]).flatten(), params, x=np.array(delay_times).flatten())
            
            # Generate tau and fit values
            tau = np.arange(0, delay_times[-1] + 5, 0.01)
            fit = result.eval(x=tau)
            
            # Plot the fit
            plt.plot(tau, fit, label=str(group), color=color)
            
            # Write fit report to file
            outtxt = f"{output_file}_{fit_type}_buildup_fit_{group}.txt"
            with open(outtxt, 'w') as file:
                file.write(result.fit_report(min_correl=0.25))
            #fit_par_tab[group] =get_biexp_fit_pars(fit_report(result))
        # Set plot labels and save the plot
        plt.xlabel("$polarization\ time$ / s")
        plt.ylabel("$NMR\ integral\ intensity$ / arb. units")
        plt.legend()
        outfile = f"{output_file}_{fit_type}_buildup_fit.pdf"
        plt.savefig(outfile, dpi='figure', format="pdf")
        plt.close()
        outtxt = f"{output_file}_{fit_type}_buildup_fit_tab.txt"
        with open(outtxt,'w') as file:
            if fit_type =="biexponential":
                file.write("Peak\tA1 / a.u. \tT1 / s\tA2 / a.u.\tT2 / s\tA1/sqrt(T1)\tA2/sqrt(T2)\n")
                for keys in fit_par_tab:   
                    file.write("\t".join([keys,fit_par_tab[keys]]))
                    file.write("\n")
            if fit_type =="monoexponential":
                file.write("Peak\tA1 / a.u. \tT1 / s\tA1/sqrt(T1)\n")
                for keys in fit_par_tab:   
                    file.write("\t".join([keys,fit_par_tab[keys]]))
                    file.write("\n")
            


    # Define output file name with "_integral_values.txt" suffix
    output_filename = f"{output_file}_integral_values.txt"

    # Open the file in write mode
    with open(output_filename, 'w') as file:
        # Write the delay times in a tab-separated format
        file.write("\t" + "\t".join(map(str, delay_times)) + "\n")

        # Write the integral values for each carbon group
        for group in carbon_groups:
            file.write(
                group + "\t" + "\t".join(map(str, integral[group])) + "\n")
