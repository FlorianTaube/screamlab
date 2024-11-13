import numpy as np
from scipy.integrate import odeint
import lmfit
import matplotlib.pyplot as plt


# Solomon equations as a system of ODEs
def solomon_system(y, t, rho_h, rho_c, sigma_HC, P_h0, P_c0):
    P_h, P_c = y
    dP_h_dt = -1 / rho_h * (P_h - P_h0) - 1 / sigma_HC * (P_c - P_c0)
    dP_c_dt = -1 / rho_c * (P_c - P_c0) - 1 / sigma_HC * (P_h - P_h0)
    return [dP_h_dt, dP_c_dt]


# Function to integrate the system and return the population of P_c over time
def solve_system(params, t, P_h_initial, P_c_initial):
    rho_h = params['rho_h']
    rho_c = params['rho_c']
    sigma_HC = params['sigma_HC']
    P_h0 = params['P_h0']
    P_c0 = params['P_c0']

    # Initial conditions
    y0 = [P_h_initial, P_c_initial]

    # Integrating the system of ODEs
    solution = odeint(solomon_system, y0, t, args=(rho_h, rho_c, sigma_HC, P_h0, P_c0))
    P_c_solution = solution[:, 1]  # We are interested in the P_c population

    return P_c_solution


# Define the model for lmfit
def solomon_model(params, t, P_h_initial, P_c_initial):
    P_c_solution = solve_system(params, t, P_h_initial, P_c_initial)
    return P_c_solution


# Function to perform the fitting and plot the results
def fit_and_plot_spectrum(t, experimental_data, P_h_initial, P_c_initial):
    # Initial guess for the parameters
    init_params = lmfit.Parameters()
    init_params.add('rho_h', value=3, min=0 , max=50)
    init_params.add('rho_c', value=17, min=0, max=50)
    init_params.add('sigma_HC', value=1.0, min=0, max=50)
    init_params.add('P_h0', value=0.5, min=-5000, max=0)
    init_params.add('P_c0', value=0.5, min=0,max=experimental_data[-1]*2)

    # Define the objective function (residuals between model and experimental data)
    def objective(params, t, experimental_data, P_h_initial, P_c_initial):
        model_data = solomon_model(params, t, P_h_initial, P_c_initial)
        weights = np.ones_like(experimental_data)
        weights[:4] = 1
        return (model_data - experimental_data)


    # Perform the fitting
    result = lmfit.minimize(objective, init_params, args=(t, experimental_data, P_h_initial, P_c_initial))

    # Print the fit results
    lmfit.report_fit(result)

    # Generate the fitted model data
    tau = np.linspace(0,33,1000)
    fitted_data = solomon_model(result.params, tau, P_h_initial, P_c_initial)

    # Plot experimental vs fitted data
    plt.figure(figsize=(10, 6))
    plt.plot(t, experimental_data, 'o', label='Experimental Data', markersize=5)
    plt.plot(tau, fitted_data, '-', label='Fitted Model', linewidth=2)
    plt.xlabel('Time (t)')
    plt.ylabel('P_c Population (Intensity)')
    plt.title('Spectrum Simulation Using Solomon Equations')
    plt.legend()
    plt.grid(True)
    plt.show()

    return result


# Generate synthetic biexponential buildup data
def generate_biexponential_data(t, A, B, k1, k2):
    return A * (1 - np.exp(-k1 * t)) + B * (1 - np.exp(-k2 * t))


# Example usage
t = np.linspace(0, 10, 10)  # Time points
experimental_data = generate_biexponential_data(t, A=1.0, B=0.5, k1=0.8, k2=0.2)  # Biexponential buildup data

experimental_data = np.array([ 49.43844, 132.79004, 348.00212, 832.11034, 1757.08512, 3043.44926, 4275.45936])
t = np.array([ 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])

P_h_initial = 0  # Initial value of P_h
P_c_initial = 0  # Initial value of P_c

# Fit the model to the experimental data and plot the results
result = fit_and_plot_spectrum(t, experimental_data, P_h_initial, P_c_initial)
