from screamlab import settings, dataset
import sys


props = settings.Properties()
props.prefit = False
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "individual"
props.output_folder = (
    rf"C:\Users\Florian Taube\Desktop\SFosF_Fosfomycin\102_P"
)
props.path_to_experiment = rf"F:\ssNMR\20260107_SFosF_Fosfomycin"
props.procno = 102
props.expno = [52, 53, 54, 55, 56, 58, 59, 60, 62, 63, 64, 65]
props.subspec = [-20, 200]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    -193,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    -150,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    -106,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    -63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    -20,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.start_analysis()

props = settings.Properties()
props.prefit = False
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "individual"
props.output_folder = (
    rf"C:\Users\Florian Taube\Desktop\SFosF_Fosfomycin\103_P"
)
props.path_to_experiment = rf"F:\ssNMR\20260107_SFosF_Fosfomycin"
props.procno = 103
props.expno = [52, 53, 54, 55, 56, 58, 59, 60, 62, 63, 64, 65]
props.subspec = [-20, 200]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    -193,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    -150,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    -106,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    -63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    -20,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.start_analysis()
import sys

sys.exit()

props = settings.Properties()
props.prefit = False
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "individual"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SFosF_Fosfomycin\102"
props.path_to_experiment = rf"F:\ssNMR\20260107_SFosF_Fosfomycin"
props.procno = 102
props.expno = [22, 34]
props.subspec = [-20, 200]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    -6,
    peak_sign="+",
)
ds.add_peak(2, peak_sign="+")
ds.add_peak(
    15,
    peak_sign="+",
)
ds.add_peak(55, peak_sign="+")
ds.add_peak(
    64,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    73,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.start_analysis()

props = settings.Properties()
props.prefit = False
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "individual"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SFosF_Fosfomycin\103"
props.path_to_experiment = rf"F:\ssNMR\20260107_SFosF_Fosfomycin"
props.procno = 103
props.expno = [22, 34]
props.subspec = [-20, 200]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    -6,
    peak_sign="-",
)
ds.add_peak(2, peak_sign="-")
ds.add_peak(
    15,
    peak_sign="-",
)
ds.add_peak(55, peak_sign="-")
ds.add_peak(64, peak_sign="-")
ds.add_peak(73, peak_sign="-")
ds.start_analysis()

import sys

sys.exit()
