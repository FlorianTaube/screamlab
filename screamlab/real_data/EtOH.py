from screamlab import settings, dataset
import sys


props = settings.Properties()
props.prefit = False
props.buildup_types = [
    "exponential",
    "biexponential",
    "exponential_with_offset",
    "biexponential_with_offset",
]
props.spectrum_fit_type = "individual"
props.output_folder = rf"F:\Dokumente\Projekte\SCREAM_DNP_EtOH_and_ILT\screamlab_output\2-13C-EtOH"
props.path_to_experiment = rf"F:\ssNMR\20260317_2-13C_EtOH"
props.procno = 103
props.expno = [20, 30]
props.subspec = [-25, 100]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    73,
    peak_sign="-",
)
ds.add_peak(
    63,
    peak_sign="-",
)
ds.add_peak(
    19,
    peak_sign="-",
)

ds.start_analysis()


props = settings.Properties()
props.prefit = False
props.buildup_types = [
    "exponential",
    "biexponential",
    "exponential_with_offset",
    "biexponential_with_offset",
]
props.spectrum_fit_type = "individual"
props.output_folder = rf"F:\Dokumente\Projekte\SCREAM_DNP_EtOH_and_ILT\screamlab_output\1-13C-EtOH"
props.path_to_experiment = rf"F:\ssNMR\20260318_1-13C_EtOH"
props.procno = 103
props.expno = [20, 30]
props.subspec = [-25, 100]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    73,
    peak_sign="-",
)
ds.add_peak(
    63,
    peak_sign="-",
)
ds.add_peak(
    59,
    peak_sign="-",
)

ds.start_analysis()

props = settings.Properties()
props.prefit = False
props.buildup_types = [
    "exponential",
    "biexponential",
    "exponential_with_offset",
    "biexponential_with_offset",
]
props.spectrum_fit_type = "individual"
props.output_folder = rf"F:\Dokumente\Projekte\SCREAM_DNP_EtOH_and_ILT\screamlab_output\13C2-EtOH"
props.path_to_experiment = rf"F:\ssNMR\20260316_13C2_EtOH"
props.procno = 103
props.expno = [20, 30]
props.subspec = [-25, 100]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    73,
    peak_sign="-",
)
ds.add_peak(
    63,
    peak_sign="-",
)
ds.add_peak(
    59,
    peak_sign="-",
)
ds.add_peak(
    19,
    peak_sign="-",
)

ds.start_analysis()

props = settings.Properties()
props.prefit = False
props.buildup_types = [
    "exponential",
    "biexponential",
    "exponential_with_offset",
    "biexponential_with_offset",
]
props.spectrum_fit_type = "global"
props.output_folder = rf"F:\Dokumente\Projekte\SCREAM_DNP_EtOH_and_ILT\screamlab_output\13C_EtOH_Mixture"
props.path_to_experiment = rf"F:\ssNMR\20260319_13C_EtOH_Mixture"
props.procno = 103
props.expno = [20, 30]
props.subspec = [-25, 100]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    73,
    peak_sign="-",
)
ds.add_peak(
    63,
    peak_sign="-",
)
ds.add_peak(
    59,
    peak_sign="-",
)
ds.add_peak(
    19,
    peak_sign="-",
)

ds.start_analysis()
