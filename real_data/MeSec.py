from screamlab import settings, dataset
import sys

props = settings.Properties()
props.prefit = False
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "numint"
props.output_folder = (
    rf"C:\Users\Florian Taube\Desktop\SMeSecF_MeSec_na\102_Se"
)
props.path_to_experiment = rf"F:\ssNMR\20260106_CE_S1_MeSec"
props.procno = 102
props.expno = [40, 46]


ds = dataset.Dataset()
ds.props = props
ds.add_peak(0, integration_range=[130, 160])

ds.start_analysis()


props = settings.Properties()
props.prefit = False
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "numint"
props.output_folder = (
    rf"C:\Users\Florian Taube\Desktop\SMeSecF_MeSec_na\103_Se"
)
props.path_to_experiment = rf"F:\ssNMR\20260106_CE_S1_MeSec"
props.procno = 103
props.expno = [40, 46]


ds = dataset.Dataset()
ds.props = props
ds.add_peak(0, integration_range=[130, 160])

ds.start_analysis()

props = settings.Properties()
props.prefit = False
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "individual"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SMeSecF_MeSec_na\102"
props.path_to_experiment = rf"F:\ssNMR\20260106_CE_S1_MeSec"
props.procno = 102
props.expno = [24, 34]
props.subspec = [0, 200]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    173,
    peak_sign="+",
    peak_label="CO",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    73,
    peak_sign="+",
    peak_label="glycerol2",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    63,
    peak_sign="+",
    peak_label="glycerol1",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    52,
    peak_label="alpha",
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    24,
    peak_sign="+",
    peak_label="gamma",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    6,
    peak_sign="+",
    peak_label="beta",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    3,
    peak_sign="+",
    peak_label="capplusglycerol",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.start_analysis()


props = settings.Properties()
props.prefit = False
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "individual"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SMeSecF_MeSec_na\103"
props.path_to_experiment = rf"F:\ssNMR\20260106_CE_S1_MeSec"
props.procno = 103
props.expno = [22, 34]
props.subspec = [0, 200]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    173,
    peak_sign="-",
    peak_label="CO",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    73,
    peak_sign="-",
    peak_label="glycerol2",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    63,
    peak_sign="-",
    peak_label="glycerol1",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    52,
    peak_label="alpha",
    peak_sign="+",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    24,
    peak_sign="-",
    peak_label="gamma",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    6,
    peak_sign="-",
    peak_label="beta",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.add_peak(
    3,
    peak_sign="-",
    peak_label="capplusglycerol",
    line_broadening={"gamma": {"max": 3}, "sigma": {"max": 3}},
)
ds.start_analysis()

###############################
