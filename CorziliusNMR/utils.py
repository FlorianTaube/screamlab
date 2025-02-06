from scipy.special import wofz
import lmfit
import numpy as np
import matplotlib.pyplot as plt


class Fitter:

    def __init__(self, dataset):
        self.dataset = dataset


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
                        intensity_values.append(
                            spectrum.peaks[peak_nr].area_under_peak["global"]
                        )
                        # intensity_values.append(spectrum.peaks[
                        # peak_nr].hight['y_val'])
                    time_values.append(spectrum.tbup)
                self.y_val = np.array(intensity_values)
                self.x_val = np.array(time_values)
                self.peak_label = peak.peak_label
                self.fit_buildup()
            self.dataset._exp_fit.update(
                {self.fitting_type: self.fit_results}
            )

    def fit_buildup(self):
        """
        Fit the buildup model to the current x and y values.
        """

        model = lmfit.models.ExpressionModel(self.get_expression_model())
        param_dict = self.get_param_dict()
        num_samples = 50 * len(param_dict)
        params = model.make_params(**param_dict)
        param_names = list(param_dict.keys())
        bounds = np.array(
            [
                [param_dict[key]["min"], param_dict[key]["max"]]
                for key in param_names
            ]
        )

        lhs_samples = lhs(len(param_names), samples=num_samples)
        sampled_params = (
            lhs_samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        )
        best_result = None
        best_chisq = np.inf
        skipped_fits = 0
        for sample_idx, sample in enumerate(sampled_params):
            try:
                params = model.make_params()
                for i, param_name in enumerate(param_names):
                    params[param_name].set(
                        value=sample[i],
                        min=param_dict[param_name]["min"],
                        max=param_dict[param_name]["max"],
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
            "x_axis": self.x_val,
            "y_axis": self.y_val,
            "fit_report": best_result.fit_report(),
            "grouped_params": result.params,
            "result": best_result,
        }
        print(best_result.rsquared)
        self.fit_results.update(
            {"_".join(self.peak_label.split("_")[0:4]): result_dict}
        )

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
            "A1": (
                dict(value=10, min=0, max=self.y_val[-1] * 3)
                if self.y_val[-1] > 0
                else dict(value=10, max=0, min=self.y_val[-1] * 2)
            ),
            "A2": (
                dict(value=10, min=0, max=self.y_val[-1] * 3)
                if self.y_val[-1] > 0
                else dict(value=10, max=0, min=self.y_val[-1] * 2)
            ),
            "x1": dict(value=5, min=0, max=self.x_val[-1] * 2),
            "x2": dict(value=self.x_val[-3], min=0, max=self.x_val[-1] * 2),
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
            "A1": (
                dict(value=10, min=0, max=self.y_val[-1] * 3)
                if self.y_val[-1] > 0
                else dict(value=10, max=0, min=self.y_val[-1] * 2)
            ),
            "A2": (
                dict(value=10, min=0, max=self.y_val[-1] * 3)
                if self.y_val[-1] > 0
                else dict(value=10, max=0, min=self.y_val[-1] * 2)
            ),
            "x1": dict(value=5, min=0, max=self.x_val[-1] * 2),
            "x2": dict(value=self.x_val[-3], min=0, max=self.x_val[-1] * 2),
            "x0": dict(value=0, min=-1.5, max=1),
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
            "A1": (
                dict(value=10, min=0, max=self.y_val[-1] * 10)
                if self.y_val[-1] > 0
                else dict(value=10, max=0, min=self.y_val[-1] * 10)
            ),
            "x1": dict(value=5, min=0, max=self.x_val[-1] * 2),
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
            "A1": (
                dict(value=10, min=0, max=self.y_val[-1] * 10)
                if self.y_val[-1] > 0
                else dict(value=10, max=0, min=self.y_val[-1] * 10)
            ),
            "x1": dict(value=5, min=0, max=self.x_val[-1] * 2),
            "x0": dict(value=0, min=-1.5, max=1),
        }


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
    return (
        amplitude
        * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        / (sigma * np.sqrt(2 * np.pi))
    )


def lorentz_profile(x, center, gamma, amplitude):
    return (
        amplitude
        * (gamma**2 / ((x - center) ** 2 + gamma**2))
        / (np.pi * gamma)
    )


def generate_subspectrum(experiment, peak_center, offset):
    closest_index = (
        np.abs(experiment.x_axis - (int(peak_center) - offset)).argmin(),
        np.abs(experiment.x_axis - (int(peak_center) + offset)).argmin(),
    )
    return experiment.y_axis[closest_index[1] : closest_index[0]]


def generate_subspectrum_2(x_axis, return_axis, max, min, offset):
    x_axis = np.array(x_axis)
    closest_index = (
        np.abs(x_axis - (int(min) - offset)).argmin(),
        np.abs(x_axis - (int(max) + offset)).argmin(),
    )
    return return_axis[closest_index[1] : closest_index[0]]


def fwhm_gaussian(sigma):
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def fwhm_lorentzian(gamma):
    return 2 * gamma


def fwhm_voigt(sigma, gamma):
    return 0.5346 * (2 * gamma) + np.sqrt(
        0.2166 * (2 * gamma) ** 2 + 4 * np.log(2) * sigma**2
    )
