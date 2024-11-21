import sys

import lmfit
import numpy as np
from collections import defaultdict
from scipy.special import wofz
import matplotlib.pyplot as plt
class Fitter:

    def __init__(self,dataset):
        self.dataset = dataset
        self.model = None
        self.result = None

    def set_model(self):
        pass

    def fit(self):
        pass


class GlobalSpectrumFitter(Fitter):
    def __init__(self,dataset):
        super().__init__(dataset)
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
                tmp_lineshape.set_init_params()
                self.lineshapes[peak.fitting_group].append(tmp_lineshape)


    def fit(self):
        for keys in self.lineshapes:
            params = lmfit.Parameters()
            for fit_peak in self.lineshapes[keys]:
                for param_name, param_attrs in fit_peak.params.items():
                    param_name = param_name.replace('.','_')
                    params.add(param_name, **param_attrs)
            result = ""
            for nr,irgendwas in enumerate(params):#TODO Make Nicer
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




class BuildupFitter():

    def __init__(self,dataset):
        self.dataset = dataset
        self.x_val = None
        self.y_val = None

    def perform_fit(self):
        for peak_nr, peak in enumerate(self.dataset.spectra[0].peaks):
            time_values = []
            intensity_values = []
            for spectrum in self.dataset.spectra:
                intensity_values.append(spectrum.peaks[peak_nr].area_under_peak[
                                      'global'])
                time_values.append(spectrum.tbup)
            self.y_val = np.array(intensity_values)
            self.x_val = np.array(time_values)
            self.fit_buildup()

    def fit_buildup(self):
        pass

class BiexpFitter(BuildupFitter):
    def __init__(self,dataset):
        super().__init__(dataset)

    def perform_fit(self):
        super().perform_fit()

    def fit_buildup(self):
        model = lmfit.models.ExpressionModel("A1*(1-exp(-(x)/x1))+A2*(1-exp("
                                             "-(x)/x2))")
        param_dict =  {'A1': dict(value=10),
            'A2': dict(value=0,vary=True),
            'x1':dict(value=5, min=0),
            'x2':dict(value=0, min=0,vary=True)}
        params = model.make_params(**param_dict)
        result = model.fit(self.y_val,params,x=(self.x_val))
        plt.plot(self.x_val,self.y_val,"o")
        plt.plot(self.x_val,result.best_fit)
        #plt.show()
        plt.close()
        #print(result.fit_report())


class ExpFitter(BuildupFitter):
    def __init__(self,dataset):
        super().__init__()

class StrechedFitter(Fitter):
    def __init__(self,dataset):
        super().__init__()

class Lineshape():
    def __init__(self,spectrum,peak):
        self.x_axis = spectrum.x_axis
        self.y_axis = spectrum.y_axis
        self.peak = peak
        self.params = None
        pass

    def set_init_params(self):
        pass

    def peak_calculator(self, params, spectrum_nr, x_data):
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
        f"{self.peak.peak_label}_amplitude": dict(value=200, min=0) if
        self.peak.sign == "+" else dict(value=-200, max=0),
        f"{self.peak.peak_label}_center": dict(value=self.peak.hight['x_val'],
                                               min=self.peak.hight['x_val'] - 3,
                                                max=self.peak.hight['x_val']
                                                    + 3),
        f"{self.peak.peak_label}_sigma": dict(value=1.5, min=1, max=2),
        f"{self.peak.peak_label}_gamma": dict(value=0.3, min=0, max=1)}

        pass

    def peak_calculator(self, params, spectrum_nr, x_data):
        amp, cen, gam, sig = self.get_correct_param_set(params,
                                                    self.peak.peak_label)
        return voigt_profile(x_data, cen, sig, gam, amp)

    def sim_spectrum(self,result,fitting_type):
        amp, cen, gam, sig = self.get_correct_param_set(result.params,
                                                    self.peak.peak_label.replace(".","_"))
        simulated_spectrum = voigt_profile(self.x_axis, cen, sig, gam, amp)
        parameter = {"amp":amp,"cen":cen,"sig":sig,"gam":gam}
        self.peak.fitting_parameter = self.peak.fitting_parameter | {
            fitting_type:parameter}
        self.peak.simulated_peak = self.peak.simulated_peak | {fitting_type:simulated_spectrum}
        self.peak.area_under_peak = self.peak.area_under_peak | {fitting_type:np.trapz(simulated_spectrum)}
        self.peak.fitting_report.update({fitting_type: lmfit.fit_report(result)})
        plt.plot(self.x_axis,self.y_axis)
        plt.plot(self.x_axis,simulated_spectrum)
        #plt.show()
        plt.close()
        #print(lmfit.fit_report(result))

    def get_correct_param_set(self,params,peak_label):
        amp = cen = gam = sig = None
        for key, value in params.items():
            if peak_label.replace('.','_') in key:
                if "amplitude" in key:
                    amp = value
                if "center" in key:
                    cen = value
                if "sigma" in key:
                    sig = value
                if "gamma" in key:
                    gam = value
        return amp, cen, gam, sig

class Gauss(Lineshape):
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
        f"{self.peak.peak_label}_sigma": dict(value=1.5, min=1, max=2)}

        pass

    def peak_calculator(self, params, spectrum_nr, x_data):
        amp, cen, sig = self.get_correct_param_set(params,
                                                    self.peak.peak_label)
        return gauss_profile(x_data, cen, sig, amp)

    def sim_spectrum(self,result,fitting_type):
        amp, cen, sig = self.get_correct_param_set(result.params,
                                                    self.peak.peak_label.replace(".","_"))
        simulated_spectrum = gauss_profile(self.x_axis, cen, sig, amp)
        parameter = {"amp":amp,"cen":cen,"sig":sig}
        self.peak.fitting_parameter = self.peak.fitting_parameter | {
            fitting_type:parameter}
        self.peak.simulated_peak = self.peak.simulated_peak | {fitting_type:simulated_spectrum}
        self.peak.area_under_peak = self.peak.area_under_peak | {fitting_type:np.trapz(simulated_spectrum)}
        plt.plot(self.x_axis,self.y_axis)
        plt.plot(self.x_axis,simulated_spectrum)
        plt.show()
        #print(lmfit.fit_report(result))

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

    def peak_calculator(self, params, spectrum_nr, x_data):
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
        plt.plot(self.x_axis,self.y_axis)
        plt.plot(self.x_axis,simulated_spectrum)
        plt.show()
        #print(lmfit.fit_report(result))

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



def peak_objective(params,lineshapes):
    residual = [[] for _ in range(len(lineshapes))]
    for spectrum_nr,spectrum in enumerate(lineshapes):
        residual[spectrum_nr] = np.zeros_like(spectrum.y_axis)
        residual[spectrum_nr] = spectrum.y_axis - spectrum.peak_calculator(
            params, spectrum_nr, spectrum.x_axis)
    return np.concatenate(residual)

def voigt_profile(x, center, sigma, gamma, amplitude):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

import numpy as np

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
