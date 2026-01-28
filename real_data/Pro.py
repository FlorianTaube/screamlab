from screamlab import settings, dataset
import sys


props = settings.Properties()
props.prefit = False
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SProF_Pro-na\102"
props.path_to_experiment = (
    rf"F:\ssNMR\20260109_1M_HN-P-OH_na_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
props.procno = 102
props.expno = [24, 34]
props.subspec = [0, 100]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    29,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    35,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    51,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    66,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    77,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.start_analysis()
sys.exit()
props = settings.Properties()
props.prefit = False
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SProF_Pro-na\103"
props.path_to_experiment = (
    rf"F:\ssNMR\20260109_1M_HN-P-OH_na_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
props.procno = 103
props.expno = [24, 34]
props.subspec = [0, 200]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    29,
    peak_sign="-",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    35,
    peak_sign="-",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    51,
    peak_sign="-",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    65,
    peak_sign="-",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)

ds.start_analysis()
sys.exit()
