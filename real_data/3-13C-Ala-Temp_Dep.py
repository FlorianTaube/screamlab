from screamlab import settings, dataset
import sys

props = settings.Properties()
props.prefit = False
props.spectrum_for_prefit = -5
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = (
    rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_7kHz_110K_103"
)
props.path_to_experiment = rf"F:\ssNMR\20260119_S3_Cle_Ala_7kHz_110K"
props.procno = 102
props.expno = [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
props.subspec = [0, 100]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    3,
    peak_sign="+",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    17,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    52,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    73,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    177,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.start_analysis()

props = settings.Properties()
props.prefit = False
props.spectrum_for_prefit = -1
props.buildup_types = ["exponential"]
props.spectrum_fit_type = "global"
props.output_folder = (
    rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_7kHz_110K_103"
)
props.path_to_experiment = rf"F:\ssNMR\20260119_S3_Cle_Ala_7kHz_110K"
props.procno = 103
props.expno = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
props.subspec = [0, 200]


ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    3,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    17,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    51,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    177,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.start_analysis()

sys.exit()


props = settings.Properties()
props.prefit = False
props.spectrum_for_prefit = -5
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = (
    rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_7kHz_100K_103"
)
props.path_to_experiment = rf"F:\ssNMR\20260119_S3_Cle_Ala_7kHz_100K"
props.procno = 102
props.expno = [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
props.subspec = [0, 100]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    3,
    peak_sign="+",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    17,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    52,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    73,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    177,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.start_analysis()

props = settings.Properties()
props.prefit = False
props.spectrum_for_prefit = -1
props.buildup_types = ["exponential"]
props.spectrum_fit_type = "global"
props.output_folder = (
    rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_7kHz_100K_103"
)
props.path_to_experiment = rf"F:\ssNMR\20260119_S3_Cle_Ala_7kHz_100K"
props.procno = 103
props.expno = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
props.subspec = [0, 200]


ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    3,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    17,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    51,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    177,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.start_analysis()
