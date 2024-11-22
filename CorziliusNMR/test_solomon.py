import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters (you can adjust these values based on your system)
R_e_to_p = 0.1  # Electron to proton polarization transfer rate
R_e_to_c = 0.05  # Electron to carbon polarization transfer rate
sigma_pc = 0.02  # Cross-relaxation rate from proton to carbon
sigma_cp = 0.02  # Cross-relaxation rate from carbon to proton
R_p = 0.1  # Proton relaxation rate
R_c = 0.05  # Carbon relaxation rate
S_e = 1  # Electron polarization (assumed constant)

# Define the differential equations for M_p(t) and M_c(t)
def model(t, y):
    M_p, M_c = y
    dM_p_dt = R_e_to_p * (S_e - M_p) - sigma_pc * M_p + sigma_cp * M_c - R_p * M_p
    dM_c_dt = R_e_to_c * (S_e - M_c) + sigma_pc * M_p - sigma_cp * M_c - R_c * M_c
    return [dM_p_dt, dM_c_dt]

# Time delays for the experiment
time_delays = [1, 2, 4, 8, 16]  # in seconds

# Initial conditions: M_p(0) = 0, M_c(0) = 0
initial_conditions = [0, 0]

# Solve the system for each delay time using solve_ivp
solutions = []
for delay in time_delays:
    # Use an adaptive time step solver
    sol = solve_ivp(model, [0, delay], initial_conditions, t_eval=np.linspace(0, delay, 100), method='RK45')
    solutions.append(sol)

# Plot the results for M_c(t) (carbon magnetization)
plt.figure(figsize=(8, 6))

for sol, delay in zip(solutions, time_delays):
    plt.plot(sol.t, sol.y[1], label=f'Delay time = {delay} s')

plt.xlabel('Time (s)')
plt.ylabel('Magnetization on Carbon (M_c)')
plt.title('Magnetization on Carbon vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Optionally, you can extract M_c at each delay time:
M_c_at_delays = [sol.y[1][-1] for sol in solutions]
print("Magnetization on Carbon at each delay time:")
for delay, M_c in zip(time_delays, M_c_at_delays):
    print(f"Delay time = {delay} s: M_c = {M_c:.4f}")
