import sys

import lmfit
import numpy as np
from collections import defaultdict
from scipy.special import wofz
import matplotlib.pyplot as plt
from pyDOE2 import lhs
class Fitter:

    def __init__(self,dataset):
        self.dataset = dataset
        self.model = None
        self.result = None

    def set_model(self):
        pass

    def start_prefit(self):
        pass

    def fit(self):
        pass


class GlobalSpectrumFitter(Fitter):
    def __init__(self,dataset):
        super().__init__(dataset)
        self.lineshapes = defaultdict(list)

    def start_prefit(self):
        # Extract bounds and initial values from parameters
        param_names = []
        param_bounds = []
        for peak in self.dataset.spectra[-1].peaks:
            if peak.fitting_model == "voigt":
                tmp_lineshape = Voigt(self.dataset.spectra[-1], peak)
            tmp_lineshape.set_init_params()
            self.lineshapes['prefit'].append(tmp_lineshape)

        params = lmfit.Parameters()
        for fit_peak in self.lineshapes['prefit']:
            for param_name, param_attrs in fit_peak.params.items():
                param_name = param_name.replace('.', '_')
                params.add(param_name, **param_attrs)
                # Collect parameter bounds for LHS
                if 'min' in param_attrs and 'max' in param_attrs:
                    param_names.append(param_name)
                    param_bounds.append(
                        (param_attrs['min'], param_attrs['max']))
        n_samples = 1
        lhs_samples = lhs(len(param_names), samples=n_samples)
        sampled_params = []
        for sample in lhs_samples:
            scaled_sample = []
            for i, (low, high) in enumerate(param_bounds):
                scaled_sample.append(low + sample[i] * (high - low))
            sampled_params.append(scaled_sample)
        results = []
        best_result = None
        best_chisquared = np.inf
        for sample_nr,sample in enumerate(sampled_params):
            print(f"{sample_nr+1}/{n_samples}")
            for i, value in enumerate(sample):
                params[param_names[i]].value = value

            result = lmfit.minimize(prefit_objective, params,
                                    args=(self.lineshapes['prefit'],))

            print(f"Number of function evaluations: {result.nfev}")
            if result.chisqr < best_chisquared:
                best_result = result
                best_chisquared = result.chisqr
                print(best_chisquared)

        bestfit = np.zeros(len(self.dataset.spectra[-1].y_axis))
        sim_params = []
        for key_nr,key in enumerate(best_result.params):
            sim_params.append(best_result.params[key].value)
            if (key_nr + 1) % 4 == 0:
                bestfit = bestfit + voigt_profile(self.dataset.spectra[-1].x_axis,
                                         sim_params[1],sim_params[2],
                                         sim_params[3],sim_params[0])
                plt.plot(self.dataset.spectra[-1].x_axis,voigt_profile(self.dataset.spectra[-1].x_axis,
                                         sim_params[1],sim_params[2],
                                         sim_params[3],sim_params[0]),"b")
                sim_params = []
        plt.plot(self.dataset.spectra[-1].x_axis,self.dataset.spectra[-1].y_axis)
        plt.plot(self.dataset.spectra[-1].x_axis,bestfit,"r--")
        plt.savefig(self.dataset.file_name_generator.get_prefit_pdf())
        file = self.dataset.file_name_generator.get_prefit_txt()
        with open(file, "w") as txt_file:
            txt_file.write(lmfit.fit_report(best_result))

        for peak in self.dataset.spectra[-1].peaks:
            peak.prefit_dict={
                f"{peak.peak_label}_amplitude":
                    dict(value=200, min=0,max=1e8) if
                    peak.sign == "+" else dict(value=-200, min= -1e8,max=0),
                f"{peak.peak_label}_center":
                    dict(value=
                         best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_center"].value,
                    min=best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_center"].value - 0.1,
                    max=best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_center"].value +0.1),
                f"{peak.peak_label}_sigma":
                    dict(value=best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_sigma"].value,
                         min=best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_sigma"].value -
                             best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_sigma"].value*0.05,
                         max=best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_sigma"].value +
                             best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_sigma"].value*0.05),
                f"{peak.peak_label}_gamma":
                    dict(value=best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_gamma"].value,
                         min=best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_gamma"].value -
                             best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_gamma"].value*0.05,
                         max=best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_gamma"].value +
                             best_result.params
                         [f"{peak.peak_label.replace('.', '_')}_gamma"].value*0.05),}
        self.lineshapes = defaultdict(list)








    def set_model(self):
        for spectrum in self.dataset.spectra:
            for peak in spectrum.peaks:
                if peak.fitting_model == "voigt":
                    tmp_lineshape = Voigt(spectrum,peak)
                elif peak.fitting_model == "gauss":
                    tmp_lineshape = Gauss(spectrum,peak)
                elif peak.fitting_model == "lorentz":
                    tmp_lineshape= Lorentz(spectrum,peak)
                if peak.prefit_dict == None:
                    tmp_lineshape.set_init_params()
                else:
                    tmp_lineshape.params = peak.prefit_dict
                self.lineshapes[peak.fitting_group].append(tmp_lineshape)


    def fit(self):
        for keys in self.lineshapes:
            params = lmfit.Parameters()
            for fit_peak in self.lineshapes[keys]:
                for param_name, param_attrs in fit_peak.params.items():
                    param_name = param_name.replace('.','_')
                    params.add(param_name, **param_attrs)
            result = ""
            for nr, irgendwas in enumerate(reversed(params)):#TODO Make Nicer
                if nr == 0:
                    parts = irgendwas.split('_')
                    result = '_'.join(parts[3:7])
                elif result not in irgendwas:
                    if "sigma" in irgendwas:
                        tmp1 = irgendwas.split('_')
                        tmp2 = "_".join(tmp1[0:3])
                        tmp = f"{tmp2}_{result}_sigma"
                        params[irgendwas].expr = tmp
                    elif "gamma" in irgendwas:
                        tmp1 = irgendwas.split('_')
                        tmp2 = "_".join(tmp1[0:3])
                        tmp = f"{tmp2}_{result}_gamma"
                        params[irgendwas].expr = tmp
                    elif "center" in irgendwas:
                        tmp1 = irgendwas.split('_')
                        tmp2 = "_".join(tmp1[0:3])
                        tmp = f"{tmp2}_{result}_center"
                        params[irgendwas].expr = tmp
            self.result = lmfit.minimize(peak_objective,params,
                                         args=(self.lineshapes[keys],))
            for voigt in self.lineshapes[keys]:
                voigt.sim_spectrum(self.result,"global")



#TODO add rest of module to docu
"""
utils.py
=================

This module provides tools for fitting buildup curves from spectral data using
various models including exponential, biexponential, and their variants with offsets.

Classes
-------
BuildupFitter
BiexpFitter
BiexpFitterWithOffset
ExpFitter
ExpFitterWithOffset

"""

import numpy as np
import lmfit

class BuildupFitter:
    """
    Base class for fitting buildup curves from spectral data.

    Attributes
    ----------
    dataset : object
        The dataset containing spectral data to be fitted.
    x_val : numpy.ndarray or None
        The time values used for fitting.
    y_val : numpy.ndarray or None
        The intensity values used for fitting.
    peak_label : str or None
        Label of the current peak being fitted.
    fitting_type : str or None
        Type of fitting performed.
    fit_results : dict
        Dictionary containing results of the fitting process.

    Methods
    -------
    perform_fit():
        Performs fitting for all peaks in the dataset.
    fit_buildup():
        Fits the buildup model to the current x and y values.
    get_expression_model():
        Returns the mathematical expression of the model. Must be implemented by subclasses.
    get_param_dict():
        Returns the initial parameter dictionary for the model. Must be implemented by subclasses.
    """

    def __init__(self, dataset):
        """
        Initialize the BuildupFitter with a dataset.

        Parameters
        ----------
        dataset : object
            Dataset containing the spectral data to be fitted.
        """
        self.dataset = dataset
        self.x_val = None
        self.y_val = None
        self.peak_label = None
        self.fitting_type = None
        self.fit_results = dict()

    def perform_fit(self):
        """
        Perform fitting for all peaks in the dataset.
        """
        for peak_nr, peak in enumerate(self.dataset.spectra[0].peaks):
            for buildup_type in self.dataset.spectrum_fitting_type:
                time_values = []
                intensity_values = []
                for spectrum in self.dataset.spectra:
                    if buildup_type == "global":
                        intensity_values.append(spectrum.peaks[
                                  peak_nr].area_under_peak['global'])
                        #intensity_values.append(spectrum.peaks[
                         # peak_nr].hight['y_val'])
                    time_values.append(spectrum.tbup)
                self.y_val = np.array(intensity_values)
                self.x_val = np.array(time_values)
                self.peak_label = peak.peak_label
                self.fit_buildup()
            self.dataset._exp_fit.update({self.fitting_type: self.fit_results})

    def fit_buildup(self):
        """
        Fit the buildup model to the current x and y values.
        """

        model = lmfit.models.ExpressionModel(self.get_expression_model())
        param_dict = self.get_param_dict()
        num_samples = 50 * len(param_dict)
        params = model.make_params(**param_dict)
        param_names = list(param_dict.keys())
        bounds = np.array([[param_dict[key]['min'], param_dict[key]['max']]
                           for key in param_names])

        lhs_samples = lhs(len(param_names), samples=num_samples)
        sampled_params = lhs_samples * (bounds[:, 1] -  bounds[:,
                                                        0]) + bounds[:, 0]
        best_result = None
        best_chisq = np.inf
        skipped_fits = 0
        for sample_idx, sample in enumerate(sampled_params):
            try:
                params = model.make_params()
                for i, param_name in enumerate(param_names):
                    params[param_name].set(
                        value=sample[i],
                        min=param_dict[param_name]['min'],
                        max=param_dict[param_name]['max']
                    )
                result = model.fit(self.y_val, params, x=self.x_val)
                if not np.isfinite(result.chisqr):
                    raise ValueError("Chi-squared value is NaN or Inf.")
                if result.chisqr < best_chisq:
                    best_chisq = result.chisqr
                    best_result = result
            except Exception as e:
                skipped_fits += 1
                print(f"Skipping sample {sample_idx} due to error: {e}")
        print(f"Total skipped fits: {skipped_fits}/{num_samples}")
        result_dict = {
            'x_axis': self.x_val,
            'y_axis': self.y_val,
            'fit_report': best_result.fit_report(),
            'params': result.params,
            'result': best_result
        }
        self.fit_results.update({"_".join(self.peak_label.split("_")[0:4]): result_dict})

    def get_expression_model(self):
        """
        Return the mathematical expression for the model.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_param_dict(self):
        """
        Return the initial parameter dictionary for the model.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class BiexpFitter(BuildupFitter):
    """
    Class for fitting biexponential models to buildup data.
    """

    def __init__(self, dataset):
        """
        Initialize the BiexpFitter with a dataset.
        """
        super().__init__(dataset)
        self.fitting_type = "biexponential"

    def get_expression_model(self):
        """
        Return the mathematical expression for the biexponential model.
        """
        return "A1*(1-exp(-(x)/x1))+A2*(1-exp(-x/x2))"

    def get_param_dict(self):
        """
        Return the initial parameter dictionary for the biexponential model.
        """
        return {
            'A1': dict(value=10, min=0, max=self.y_val[-1]*3) if self.y_val[-1]
                    > 0 else dict(value=10, max=0, min=self.y_val[-1]*2),
            'A2': dict(value=10, min=0, max=self.y_val[-1]*3) if self.y_val[-1]
                    > 0 else dict(value=10, max=0, min=self.y_val[-1]*2),
            'x1': dict(value=5, min=0, max=self.x_val[-1]*2),
            'x2': dict(value=self.x_val[-3], min=0,max=self.x_val[-1]*2),
            }


class BiexpFitterWithOffset(BuildupFitter):
    """
    Class for fitting biexponential models with offsets to buildup data.
    """

    def __init__(self, dataset):
        """
        Initialize the BiexpFitterWithOffset with a dataset.
        """
        super().__init__(dataset)
        self.fitting_type = "biexponential_with_offset"

    def get_expression_model(self):
        """
        Return the mathematical expression for the biexponential model with offset.
        """
        return "A1*(1-exp(-(x-x0)/x1))+A2*(1-exp(-(x-x0)/x2))"

    def get_param_dict(self):
        """
        Return the initial parameter dictionary for the biexponential model with offset.
        """

        return {
            'A1': dict(value=10, min=0, max=self.y_val[-1]*3) if self.y_val[-1]
                    > 0 else dict(value=10, max=0, min=self.y_val[-1]*2),
            'A2': dict(value=10, min=0, max=self.y_val[-1]*3) if self.y_val[-1]
                    > 0 else dict(value=10, max=0, min=self.y_val[-1]*2),
            'x1': dict(value=5, min=0, max=self.x_val[-1]*2),
            'x2': dict(value=self.x_val[-3], min=0,max=self.x_val[-1]*2),
            'x0': dict(value=0, min=-1.5,max=1)
            }


class ExpFitter(BuildupFitter):
    """
    Class for fitting exponential models to buildup data.
    """

    def __init__(self, dataset):
        """
        Initialize the ExpFitter with a dataset.
        """
        super().__init__(dataset)
        self.fitting_type = "exponential"

    def get_expression_model(self):
        """
        Return the mathematical expression for the exponential model.
        """
        return "A1*(1-exp(-(x)/x1))"

    def get_param_dict(self):
        """
        Return the initial parameter dictionary for the exponential model.
        """
        return {
            'A1': dict(value=10, min=0, max=self.y_val[-1]*3) if self.y_val[-1]
                    > 0 else dict(value=10, max=0, min=self.y_val[-1]*2),
            'x1': dict(value=5, min=0,max=self.x_val[-1]*2)
        }


class ExpFitterWithOffset(BuildupFitter):
    """
    Class for fitting exponential models with offsets to buildup data.
    """

    def __init__(self, dataset):
        """
        Initialize the ExpFitterWithOffset with a dataset.
        """
        super().__init__(dataset)
        self.fitting_type = "exponential_with_offset"

    def get_expression_model(self):
        """
        Return the mathematical expression for the exponential model with offset.
        """
        return "A1*(1-exp(-(x-x0)/x1))"

    def get_param_dict(self):
        """
        Return the initial parameter dictionary for the exponential model with offset.
        """
        return {
            'A1': dict(value=10, min=0, max=self.y_val[-1]*2) if self.y_val[-1]
                    > 0 else dict(value=10, max=0, min=self.y_val[-1]*2),
            'x1': dict(value=5, min=0,max=self.x_val[-1]*2),
            'x0': dict(value=0, min=-1.5, max=1)
        }



class Lineshape():
    def __init__(self,spectrum,peak):
        self.x_axis = spectrum.x_axis
        self.y_axis = spectrum.y_axis
        self.peak = peak
        self.params = None
        pass

    def set_init_params(self):
        pass

    def peak_calculator(self, params, x_data):
        pass

    def sim_spectrum(self,result,fitting_type):
        pass

    def get_correct_param_set(self, params, peak_label):
        pass

class Voigt(Lineshape):
    def __init__(self,spectrum,peak):
        super().__init__(spectrum,peak)

    def set_init_params(self):
        self.params = {
        f"{self.peak.peak_label}_amplitude": dict(value=200, min=0,max=1e8) if
        self.peak.sign == "+" else dict(value=-200, min= -1e8,max=0),
        f"{self.peak.peak_label}_center": dict(value=self.peak.hight['x_val'],
                                               min=self.peak.hight['x_val'] -
                                                   1,
                                                max=self.peak.hight['x_val']
                                                    + 1),
        f"{self.peak.peak_label}_sigma": dict(value=1, min=0.1, max=3),
        f"{self.peak.peak_label}_gamma": dict(value=1, min=0.1, max=3)}


    def peak_calculator(self, params, x_data):
        amp, cen, gam, sig = self.get_correct_param_set(params,
                                                    self.peak.peak_label)
        return voigt_profile(x_data, cen, sig, gam, amp)

    def sim_spectrum(self,result,fitting_type):
        amp, cen, gam, sig = self.get_correct_param_set(result.params,
                                                    self.peak.peak_label.replace(".","_"))
        simulated_spectrum = voigt_profile(self.x_axis, cen, sig, gam, amp)
        parameter = {"amp":amp,"cen":cen,"sig":sig,"gam":gam}
        self.peak.fitting_parameter.update({
            fitting_type:parameter})
        self.peak.simulated_peak.update({fitting_type:simulated_spectrum})
        self.peak.area_under_peak.update( {fitting_type:np.trapz(
            simulated_spectrum)})
        self.peak.fitting_report.update({fitting_type: lmfit.fit_report(result)})

    def get_correct_param_set(self, params, peak_label):
        label_key = peak_label.replace('.', '_')
        amp = cen = gam = sig = None
        for key, value in params.items():
            if label_key in key:
                if "amplitude" in key: amp = value
                elif "center" in key: cen = value
                elif "sigma" in key: sig = value
                elif "gamma" in key: gam = value
        return amp, cen, gam, sig


class Gauss(Lineshape):
    def __init__(self,spectrum,peak):
        super().__init__(spectrum,peak)

    def set_init_params(self):
        self.params = {
        f"{self.peak.peak_label}_amplitude": dict(value=200, min=0) if
        self.peak.sign == "+" else dict(value=-200, max=0),
        f"{self.peak.peak_label}_center": dict(value=self.peak.hight['x_val'],
                                               min=self.peak.hight['x_val'] -
                                                   0.1,
                                                max=self.peak.hight['x_val']
                                                    + 0.1),
        f"{self.peak.peak_label}_sigma": dict(value=1.5, min=1, max=2)}

        pass

    def peak_calculator(self, params, x_data):
        amp, cen, sig = self.get_correct_param_set(params,
                                                    self.peak.peak_label)
        return gauss_profile(x_data, cen, sig, amp)

    def sim_spectrum(self, result, fitting_type):
        amp, cen, sig = self.get_correct_param_set(result.params,
                             self.peak.peak_label.replace(".", "_"))
        sim_spectrum = gauss_profile(self.x_axis, cen, sig, amp)
        param = {"amp": amp, "cen": cen, "sig": sig}
        self.peak.fitting_parameter |= {fitting_type: param}
        self.peak.simulated_peak |= {fitting_type: sim_spectrum}
        self.peak.area_under_peak |= {fitting_type: np.trapz(sim_spectrum)}

    def get_correct_param_set(self,params,peak_label):
        amp = cen = sig = None
        for key, value in params.items():
            if peak_label.replace('.','_') in key:
                if "amplitude" in key:
                    amp = value
                if "center" in key:
                    cen = value
                if "sigma" in key:
                    sig = value
        return amp, cen, sig


class Lorentz(Lineshape):
    def __init__(self,spectrum,peak):
        super().__init__(spectrum,peak)

    def set_init_params(self):
        self.params = {
        f"{self.peak.peak_label}_amplitude": dict(value=200, min=0) if
        self.peak.sign == "+" else dict(value=-200, max=0),
        f"{self.peak.peak_label}_center": dict(value=self.peak.hight['x_val'],
                                               min=self.peak.hight['x_val'] - 3,
                                                max=self.peak.hight['x_val']
                                                    + 3),
        f"{self.peak.peak_label}_sigma": dict(value=1.5, min=1, max=2),
        f"{self.peak.peak_label}_gamma": dict(value=0.3, min=0, max=1)}

        pass

    def peak_calculator(self, params, x_data):
        amp, cen, gam, sig = self.get_correct_param_set(params,
                                                    self.peak.peak_label)
        return lorentz_profile(x_data, cen, gam, amp)

    def sim_spectrum(self,result,fitting_type):
        amp, cen, gam = self.get_correct_param_set(result.params,
                                                    self.peak.peak_label.replace(".","_"))
        simulated_spectrum = lorentz_profile(self.x_axis, cen, gam, amp)
        parameter = {"amp":amp,"cen":cen,"gam":gam}
        self.peak.fitting_parameter = self.peak.fitting_parameter | {
            fitting_type:parameter}
        self.peak.simulated_peak = self.peak.simulated_peak | {fitting_type:simulated_spectrum}
        self.peak.area_under_peak = self.peak.area_under_peak | {fitting_type:np.trapz(simulated_spectrum)}


    def get_correct_param_set(self,params,peak_label):
        amp = cen = gam =  None
        for key, value in params.items():
            if peak_label.replace('.','_') in key:
                if "amplitude" in key:
                    amp = value
                if "center" in key:
                    cen = value
                if "gamma" in key:
                    gam = value
        return amp, cen, gam

def prefit_objective(params, lineshapes):
    residual = lineshapes[0].y_axis
    for i, voigt in enumerate(lineshapes):
        residual = residual - voigt.peak_calculator(params, voigt.x_axis)
    return residual

def peak_objective(params, lineshapes):
    peak_names = {"_".join(param.split("_")[:4]) for param in params}
    number_of_peaks = len(peak_names)
    residual = []
    for i, voigt in enumerate(lineshapes):
        calc = voigt.peak_calculator(params, voigt.x_axis)
        if i % number_of_peaks == 0:
            residual.append(voigt.y_axis - calc)
        else:
            residual[-1] -= calc
    return np.concatenate(residual)

def voigt_profile(x, center, sigma, gamma, amplitude):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def gauss_profile(x, center, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def lorentz_profile(x, center, gamma, amplitude):
    return amplitude * (gamma**2 / ((x - center)**2 + gamma**2)) / (np.pi * gamma)

def generate_subspectrum(experiment,peak_center,offset):
    closest_index = (np.abs(experiment.x_axis -
                            (int(peak_center) - offset)).argmin(),
                     np.abs(experiment.x_axis -
                            (int(peak_center) + offset)).argmin())
    return experiment.y_axis[closest_index[1]:closest_index[0]]
