from screamlab import settings, dataset
import sys

props = settings.Properties()
props.prefit = False
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SGAF_Gly-Ala-na\102"
props.path_to_experiment = rf"F:\ssNMR\20260107_SGAF_Gly-Ala-na"
props.procno = 102
props.expno = [32, 36]
props.subspec = [-20, 200]


ds = dataset.Dataset()
ds.props = props
# ds.add_peak(-6, peak_sign="+",line_broadening={"gamma":{"max":3},"sigma":{"max":3}})
# ds.add_peak(3, peak_sign="+",line_broadening={"gamma":{"max":3},"sigma":{"max":3}})
# ds.add_peak(18, peak_sign="+",line_broadening={"gamma":{"max":3},"sigma":{"max":3}})
# ds.add_peak(41, peak_sign="+",line_broadening={"gamma":{"max":3},"sigma":{"max":3}})
# ds.add_peak(52, peak_sign="+",line_broadening={"gamma":{"max":3},"sigma":{"max":3}})
# ds.add_peak(63, peak_sign="+",line_broadening={"gamma":{"max":3},"sigma":{"max":3}})
# ds.add_peak(73, peak_sign="+",line_broadening={"gamma":{"max":3},"sigma":{"max":3}})
ds.add_peak(
    168,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    181,
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.start_analysis()


props = settings.Properties()
props.prefit = False
props.spectrum_for_prefit = -1
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SGAF_Gly-Ala-na\103"
props.path_to_experiment = rf"F:\ssNMR\20260107_SGAF_Gly-Ala-na"
props.expno = [22, 36]
props.subspec = [-20, 200]


ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    18,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    41,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    52,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    73,
    peak_sign="+",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    168,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    181,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.start_analysis()
