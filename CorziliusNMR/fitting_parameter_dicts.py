def get_model_expression(fitting_type):
    if fitting_type == "Exponential":
        return "A1*(1-exp(-x/x1))"
    if fitting_type == "Exponential_with_offset":
        return "A1*(1-exp(-(x-x0)/x1))"
    elif fitting_type == "Biexponential":
        return "A1*(1-exp(-x/x1))+A2*(1-exp(-x/x2))"
    elif fitting_type == "Biexponential_with_offset":
        return "A1*(1-exp(-(x-x0)/x1))+A2*(1-exp(-(x-x0)/x2))"
    else:
        sys.exit("Enter valid fitting models")

def get_param_dict(fitting_type,position=0,sign="",prefix="",spectrum_nr=""):
    if fitting_type == "Exponential":
        return {'A1': dict(value=10),
            'x1':dict(value=5, min=0)}
    elif fitting_type == "Exponential_with_offset":
        return {'A1': dict(value=10),
            'x1':dict(value=5, min=0),
            'x0':dict(value=0,min=-10,max=10)}
    elif fitting_type == "Biexponential":
        return {'A1': dict(value=10),
            'A2': dict(value=10),
            'x1':dict(value=5, min=0),
            'x2':dict(value=5, min=0)}
    elif fitting_type == "Biexponential_with_offset":
        return {'A1': dict(value=10),
            'A2': dict(value=10),
            'x1':dict(value=5, min=0),
            'x2':dict(value=5, min=0),
            'x0':dict(value=0,min=-4,max=4)}
    elif fitting_type == "Solomon":
        return {
            'P_h': 0,
            'rho_h': 23,
            'sigma_HC': -16,
            'P_c': 0,
            'P_c0': 400,
            'P_h0': 500,
            'rho_c': 10}
    elif fitting_type == "Voigt":
        return {
            f"{prefix}_amplitude": dict(value=200,min=0) if sign == "+" else dict(
                value=-200, max=0),
            f"{prefix}_center": dict(value=position, min=position - 2,
                                     max=position + 2),
            f"{prefix}_sigma": dict(value=0.2,min=0, max=3),
            f"{prefix}_gamma": dict(value=0.2,min=0, max=3)
        }
    elif fitting_type == "global_voigt":
        return {
            f"{prefix}_{spectrum_nr}_amplitude": dict(value=200,min=0) if sign == "+" else
            dict(
                value=-200, max=0),
            f"{prefix}_{spectrum_nr}_center": dict(value=position, min=position - 2,
                                     max=position + 2),
            f"{prefix}_{spectrum_nr}_sigma": dict(value=0.2,min=0, max=3),
            f"{prefix}_{spectrum_nr}_gamma": dict(value=0.2,min=0, max=3)
        }
    else:
        sys.exit("Enter valid fitting models")

def get_param_ranges(fitting_type,delay_times,intensitys_result):
    if fitting_type == "Exponential":
        return {'A1': (0, intensitys_result[-1]*2),
            'x1':(0, delay_times[-1]*2)}
    elif fitting_type == "Exponential_with_offset":
        return {'A1': (0, intensitys_result[-1]*2),
            'x1':(0, delay_times[-1]*2),
            'x0':(-10,10)}
    elif fitting_type == "Biexponential":
        return {'A1': (0, intensitys_result[-1]*2),
                'A2': (0, intensitys_result[-1]*2),
            'x1':(0.5, delay_times[-1]*2),
            'x2':(0.5, delay_times[-1]*2)}
    elif fitting_type == "Biexponential_with_offset":
        return {'A1': (0, intensitys_result[-1]*2),
                'A2': (0, intensitys_result[-1]*2),
            'x1':(0, delay_times[-1]*2),
            'x2':(0, delay_times[-1]*2),
            'x0':(-10,10)}
    elif fitting_type == "Solomon":
        return{
            'P_c0': (0, intensitys_result[-1]*2),
            'P_h0': (intensitys_result[-1]*-2, intensitys_result[-1]*2),
            'rho_c': (0,delay_times[-1]*2),
            'rho_h': (0, delay_times[-1] * 2),
            'sigma_HC': (delay_times[-1]*-2, 0),
        }
    else:
        sys.exit("Enter valid fitting models")