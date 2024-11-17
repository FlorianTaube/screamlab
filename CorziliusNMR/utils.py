import numpy as np

class Fitter:

    def __init__(self,dataset):
        self.dataset = dataset
        self.model = None
        self.params = None

    def _set_model(self):
        pass




def generate_subspectrum(experiment,peak_center,offset):
    closest_index = (np.abs(experiment.x_axis -
                            (int(peak_center) - offset)).argmin(),
                     np.abs(experiment.x_axis -
                            (int(peak_center) + offset)).argmin())
    return experiment.y_axis[closest_index[0]:closest_index[1]]









'''
import csv
import numpy as np
from matplotlib import cm
from CorziliusNMR import fitting_parameter_dicts
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import lmfit
from lmfit.models import ExpressionModel,Model
from pyDOE import lhs
import sys
from scipy.integrate import solve_ivp,odeint
from scipy.special import wofz

def generate_csv(output_file,procno):
    return output_file+"_"+procno+".csv"

def get_delay_times_from_csv(csv_name):
    with open(csv_name, 'r') as file:
        for count, row in enumerate(csv.reader(file)):
            if count == 10:
                return [float(value.strip(' #[].')) for value in row if float(value.strip(' #[].')) != 0]

def read_xy_data_from_csv(csv_file_name):
    data = np.loadtxt(csv_file_name, delimiter=",")
    x_data = [data[:, i] for i in range(0, np.size(data, 1)) if i % 2 == 0]
    y_data = [data[:, i] for i in range(0, np.size(data, 1)) if i % 2 != 0]
    return x_data, y_data

def generate_peak_dict(peak_list, plus_minus_list):
    peak_dict = dict()
    if len(peak_list) == len(plus_minus_list):
        for peak, plus_minus in zip(peak_list, plus_minus_list):
            peak_dict[peak] = [plus_minus]
    elif len(peak_list) != len(plus_minus_list) and len(plus_minus_list) == 1:
        for peak in peak_list:
            peak_dict[peak] = [plus_minus_list[0]]
    elif len(peak_list) < len(plus_minus_list):
        for nr,peak in enumerate(peak_list):
            peak_dict[peak] = [plus_minus_list[nr]]
    return peak_dict

def check_if_autopeakpick_is_possible(plus_minus):
    return all(element == plus_minus[0] for element in plus_minus)

def check_if_plus_minus_list_contains_just_plus_and_minus(plus_minus_list):
    return True if all(element in {"+", "-"} for element in plus_minus_list) else print(
        "ERROR: All elements in the plus_minus_list must be '+' or '-'") or False

def auto_peak_pick(peak_list,plus_minus_list,x_data,y_data,xy_return="x"):
    height = 1
    multiplier = int(plus_minus_list[0] + "1")
    peaks, _ = find_peaks(multiplier * y_data, height=height,
                          distance=15)
    while len(peaks) > len(peak_list):
        height += 1
        peaks, _ = find_peaks(multiplier * y_data, height=height, distance=5)
    if xy_return == "x":
        return [int(x_data[peak]) for peak in peaks]
    else:
        return y_data[peaks[0]]

def gaussian(x, amp, center, width):
    return amp * np.exp(-((x - center) ** 2) / (2 * width ** 2))

def add_peak_label(peak_infos,procpars):
    for keys in peak_infos:
        peak_infos[keys].append(f"Peak_at_{keys}_ppm_procno_{procpars[0]}")
    for peak in peak_infos:
        peak_infos[peak][1] = peak_infos[peak][1].replace("-", "m")
    return peak_infos

def get_intensitys_from_maximum(x_data,y_data,peak_infos):
    intensitys = dict()
    for keys in peak_infos:
        intensitys[peak_infos[keys][1]] = []
        for nr,x_data_list in enumerate(x_data):
            closest_index = (np.argmin(np.abs(x_data_list - keys-4)),np.argmin(np.abs(x_data_list - keys+4)))
            intensitys[peak_infos[keys][1]].append(auto_peak_pick([keys], [peak_infos[keys][0]],
                           get_subspectrum(x_data[nr],closest_index),
                           get_subspectrum(y_data[nr],closest_index),
                           xy_return="y"))
    return intensitys

def get_subspectrum(data,closest_index):
    return data[closest_index[0]:closest_index[1]]

def calc_buildup(intensitys_result,delay_times,output_file,
                 type,fitting_type_list=["Exponential"]):
    for fitting_type in fitting_type_list:
        for idx,key in enumerate(intensitys_result):
            color = cm.viridis(idx / len(intensitys_result))
            model,params = generate_model(fitting_type)
            result = get_best_fit(model,intensitys_result[key],params,delay_times,fitting_type)
            plt.plot(delay_times,intensitys_result[key],"o",color=color,label=key.split("_procno")[0])
            tau = np.linspace(0,int(delay_times[-1]),1000)
            if fitting_type == "Solomon":
                tau = np.linspace(0, int(delay_times[-1]), 1000)
                fitted_data = solomon_model(result.params, tau, 0, 0)
                plt.plot(tau, fitted_data,color=color)
            else:
                plt.plot(tau,result.eval(x=tau),color=color)
            save_fitting_report(result,fitting_type,output_file,key,type)
        file = output_file + "_"+type+"_" + fitting_type + ".pdf"
        plt.xlabel(r"$\it{Buildup\ time}$ / s")
        plt.ylabel(r"$\it{Signal\ intensity}$ / a.u.")
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
        plt.legend()
        plt.savefig(file, format="pdf")
        #plt.show()
        plt.close()

def generate_model(fitting_type):
    param_dict = fitting_parameter_dicts.get_param_dict(fitting_type)
    if fitting_type == "Solomon":
        model = lmfit.Model(solomon_two_spin)
    else:
        model_expression = fitting_parameter_dicts.get_model_expression(fitting_type)
        model = ExpressionModel(model_expression)
    params = model.make_params(**param_dict)
    return model,params

def save_fitting_report(result,fitting_type,output_file,key,type):
    file =output_file+"_"+key+"_"+type+"_"+fitting_type+".txt"
    with open(file, "w") as file:
        if fitting_type == "Solomon":
            file.write(lmfit.fit_report(result))
        else:
            file.write((result.fit_report()))

def get_best_fit(model,intensitys_result,params,delay_times,fitting_type,
                 num_samples=100):

    lhs_samples = lhs(len(params), samples=num_samples)
    param_ranges = fitting_parameter_dicts.get_param_ranges(fitting_type,delay_times,intensitys_result)
    best_result = None
    best_residual = np.inf
    best_rsquared = 0
    for nr,sample in enumerate(lhs_samples):
        print(nr)
        print(nr) if (nr + 1) % 25 == 0 else None
        for key_nr,key in enumerate(param_ranges):
            params[key].value = sample[key_nr]*(param_ranges[key][1]-param_ranges[key][0])+param_ranges[key][0]
            if param_ranges[key][0] < param_ranges[key][1]:
                params[key].min  =param_ranges[key][0]
                params[key].max = param_ranges[key][1]
            else:
                params[key].min = param_ranges[key][1]
                params[key].max = param_ranges[key][0]
        try:

            if fitting_type == "Solomon":
                P_h_initial = 0
                P_c_initial = 0
                result = fit_and_plot_spectrum(delay_times, intensitys_result, P_h_initial, P_c_initial,params)
                fitted_data = solomon_model(result.params, delay_times, P_h_initial, P_c_initial)
                residual = abs(sum(intensitys_result - fitted_data))
                if best_residual > residual:
                    best_residual = residual
                    best_result = result
            else:
                result = model.fit(intensitys_result,params,x=delay_times)
                residual = sum(intensitys_result - result.best_fit)
                rsquared = result.rsquared
                if best_rsquared < rsquared:
                    best_result = result
                    best_rsquared = rsquared
        except:
            pass
    print(best_result)
    return best_result

def solomon_two_spin(x,  P_h0, rho_h, sigma_HC, P_c0, rho_c):
    def coupled_system(z,t):
        P_h, P_c = z
        dP_h_dt = -1/rho_h * (P_h - P_h0) - 1/sigma_HC * (P_c - P_c0)
        dP_c_dt = -1/rho_c * (P_c - P_c0) - 1/sigma_HC * (P_h - P_h0)
        return [dP_h_dt, dP_c_dt]
    initial_conditions = [0, 0]
    solution = odeint(coupled_system, initial_conditions, x)
    return solution.T[1]

def solomon_system(y, t, rho_h, rho_c, sigma_HC, P_h0, P_c0):
    P_h, P_c = y
    dP_h_dt = -1 / rho_h * (P_h - P_h0) - 1 / sigma_HC * (P_c - P_c0)
    dP_c_dt = -1 / rho_c * (P_c - P_c0) - 1 / sigma_HC * (P_h - P_h0)
    return [dP_h_dt, dP_c_dt]

def solve_system(params, t, P_h_initial, P_c_initial):
    rho_h = params['rho_h']
    rho_c = params['rho_c']
    sigma_HC = params['sigma_HC']
    P_h0 = params['P_h0']
    P_c0 = params['P_c0']
    y0 = [P_h_initial, P_c_initial]
    solution = odeint(solomon_system, y0, t, args=(rho_h, rho_c, sigma_HC, P_h0, P_c0))
    P_c_solution = solution[:, 1]
    return P_c_solution

def solomon_model(params, t, P_h_initial, P_c_initial):
    P_c_solution = solve_system(params, t, P_h_initial, P_c_initial)
    return P_c_solution

def fit_and_plot_spectrum(t, experimental_data, P_h_initial, P_c_initial,params):
    def builtup_time_objective(params, t, experimental_data, P_h_initial, P_c_initial):
        model_data = solomon_model(params, t, P_h_initial, P_c_initial)
        weights = np.ones_like(experimental_data)
        weights[:4] = 4
        return (model_data - experimental_data)*weights
    result = lmfit.minimize(builtup_time_objective, params, args=(t, experimental_data, P_h_initial, P_c_initial))
    return result


def get_intensitys_from_voigt_fittings(x_data, y_data, peak_infos,
                                       fitting_type="Voigt"):
    intensitys = dict()
    for keys in peak_infos:
        intensitys[peak_infos[keys][1]] = []
    for spectrum_nr, spectrum in enumerate(y_data):
        print(spectrum_nr)
        peak = []
        params = lmfit.Parameters()
        for key in peak_infos:
            model = Model(voigt_profile_single, prefix=f'{peak_infos[key][1]}_')
            param_dict = fitting_parameter_dicts.get_param_dict(
                fitting_type,position=key,sign=peak_infos[key][0],
                prefix=peak_infos[key][1])
            model.make_params(**param_dict)
            for param in param_dict:
                params.add(param,**param_dict[param])
            peak.append(model)
        composite_model = peak[0]
        try:
            for model in peak[1:]:
                composite_model += model
        except:
            pass
        result = composite_model.fit(y_data[spectrum_nr],params,x=x_data[spectrum_nr])
        for key in peak_infos:
            amp = result.params[f"{peak_infos[key][1]}_amplitude"].value
            cen = result.params[f"{peak_infos[key][1]}_center"].value
            sig = result.params[f"{peak_infos[key][1]}_sigma"].value
            gam = result.params[f"{peak_infos[key][1]}_gamma"].value
            intensitys[peak_infos[key][1]].append(sum(voigt_profile_single
                                                   (x_data[spectrum_nr],amp,
                                                    cen,sig,gam)))

        plt.plot(x_data[spectrum_nr],y_data[spectrum_nr], color="black")
        plt.plot(x_data[spectrum_nr],result.best_fit, linestyle="--", color="red")
    plt.plot([], [], color="black", label="Experimental")
    plt.plot([], [], linestyle="--", color="red", label="Simulation")
    plt.xlabel("chemical shift / ppm")
    plt.ylabel("signal intensity / a.u.")
    plt.legend()
    #plt.show()
    plt.close()
    return intensitys


def get_intensitys_from_global_voigt_fittings(
        x_data, y_data, peak_infos, fitting_together, output,
        fitting_type="global_voigt"):
    intensitys = {info[1]: [] for info in peak_infos.values()}
    for sublist in fitting_together:
        peak_info_sublist = generate_peak_info_sublist(peak_infos,sublist)
        param = lmfit.Parameters()
        for key in peak_info_sublist:
            param.update(generate_param_set_for_global_voigt(key,fitting_type,
                                                        peak_info_sublist,
                                                        y_data))
            param = add_global_fitting_conditions(param, y_data,
                                                  peak_info_sublist,key)

        result = lmfit.minimize(voigt_objective, param, args=(x_data, y_data))
        save_global_voigtan_deconvolution(peak_info_sublist,result,output)
        print_global_voigt_fitting_result(x_data,y_data,result,
                                          peak_info_sublist,output)
        for key in peak_info_sublist:
            intensitys[peak_info_sublist[key][1]] = \
                calculate_integral_for_each_peak(x_data,y_data,result,
                                                 peak_info_sublist,key)
    return intensitys

def calculate_integral_for_each_peak(x_data,y_data,result,peak_info_sublist,key):
    integrals = []
    for spectrum_nr, spectrum in enumerate(y_data):
        fittin_params = []
        for variable in ["amplitude","center","sigma","gamma"]:
            fittin_params.append([result.params[f"{peak_info_sublist[key][1]}"
                            f"_{spectrum_nr}_{variable}"].value])
        integrals.append(sum(voigt_profile(x_data[spectrum_nr],fittin_params[0],
                                 fittin_params[1],fittin_params[2],
                                 fittin_params[3])))
    return integrals
def generate_peak_info_sublist(peak_infos,sublist):
    dict_keys = list(peak_infos.keys())
    peak_infos_subdict = dict()
    for liste in sublist:
        peak_infos_subdict[dict_keys[liste-1]] = peak_infos[dict_keys[liste-1]]
    return peak_infos_subdict

def print_global_voigt_fitting_result(x_data,y_data,result,peak_info_sublist,
                                      output):
    for spectrum_nr, spectrum in enumerate(y_data):
        plt.plot(x_data[spectrum_nr],y_data[spectrum_nr],color='black')
        sim_spectrum = 0
        for keys in peak_info_sublist:
            fittin_params = []
            for variable in ["amplitude","center","sigma","gamma"]:
                fittin_params.append([result.params[f"{peak_info_sublist[keys][1]}"
                                f"_{spectrum_nr}_{variable}"].value])
            sim_peak = voigt_profile(x_data[spectrum_nr],fittin_params[0],
                                     fittin_params[1],fittin_params[2],
                                     fittin_params[3])
            sim_spectrum += sim_peak
        plt.plot(x_data[spectrum_nr],sim_spectrum,"r--")
    plt.plot([], [], color="black", label="Experimental")
    plt.plot([], [], linestyle="--", color="red", label="Simulation")
    plt.xlabel("chemical shift / ppm")
    plt.ylabel("signal intensity / a.u.")
    plt.legend()
    peak = "_".join(map(str, peak_info_sublist))
    file_name = f"{output}_global_voigt_deconvolution_for_peak_at_{peak}.pdf"
    plt.savefig(file_name, format="pdf")
    plt.close()

def save_global_voigtan_deconvolution(peak_info_sublist,result,output):
    peak = "_".join(map(str, peak_info_sublist))
    file_name = f"{output}_global_voigt_deconvolution_for_peak_at_{peak}.txt"
    with open(file_name, "w") as file:
        file.write(lmfit.fit_report(result))

def add_global_fitting_conditions(param, y_data,peak_info_sublist,key):
    for spectrum_nr in range(1,len(y_data)):
        for variable in ["center","sigma","gamma"]:
            param[f"{peak_info_sublist[key][1]}_{spectrum_nr}_" \
                  f"{variable}"].expr = f"{peak_info_sublist[key][1]}_0_" \
                  f"{variable}"
    return param
def generate_param_set_for_global_voigt(key,fitting_type,peak_info_sublist,
                                        y_data):
    param = lmfit.Parameters()
    for spectrum_nr,spectrum in enumerate(y_data):
        params = (fitting_parameter_dicts.get_param_dict(fitting_type,
                position=key,sign=peak_info_sublist[key][0],
                prefix=peak_info_sublist[key][1],spectrum_nr=spectrum_nr))
        for param_key in params:
            param.add(param_key,**params[param_key])
    return param

def voigt_objective(params, x_data, y_data):
        residual = np.zeros_like(y_data)
        for spectrum_nr, spectrum in enumerate(y_data):
            residual[spectrum_nr] =y_data[spectrum_nr] - voigt_dataset(
                params, spectrum_nr, x_data[spectrum_nr])
        return residual.flatten()

def voigt_dataset(params, spectrum_nr, x_data):
    amp,cen,gam,sig = get_correct_param_set(params,spectrum_nr)
    return voigt_profile(x_data,amp,cen,sig,gam)

def get_correct_param_set(params,spectrum_nr):
    amp, cen, gam, sig = [], [], [], []
    key_map = {"amplitude": amp, "center": cen, "sigma": sig, "gamma": gam}
    for key, value in params.items():
        if key.split("_")[6] == str(spectrum_nr):
            for k, lst in key_map.items():
                if k in key:
                    lst.append(value)
    return amp, cen, gam, sig

def voigt_profile_single(x, amplitude, center, sigma, gamma):
    return (amplitude*np.real(wofz((x-center+1j*gamma)/(sigma*np.sqrt(2)))))/(sigma*np.sqrt(
            2*np.pi))
def voigt_profile(x, amplitude, center, sigma, gamma):
    return sum((amplitude[number]*np.real(wofz((x-center[number]+1j*gamma[
            number])/(sigma[number]*np.sqrt(2)))))/(sigma[number]*np.sqrt(
            2*np.pi)) for number in range(len(amplitude)))

'''

