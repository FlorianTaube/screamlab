from screamlab import settings, dataset
import sys

props = settings.Properties()
props.prefit = False
props.spectrum_for_prefit = -5
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "individual"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_0.5_102"
props.path_to_experiment = rf"F:\ssNMR\20260112_SAlaF_0.5M_Ala"
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
    63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    73,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.start_analysis()

props = settings.Properties()
props.prefit = True
props.spectrum_for_prefit = -1
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_0.5_103"
props.path_to_experiment = rf"F:\ssNMR\20260112_SAlaF_0.5M_Ala"
props.procno = 103
props.expno = [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
props.subspec = [0, 100]


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
    63,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    73,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.start_analysis()

#############################################
props = settings.Properties()
props.prefit = False
props.spectrum_for_prefit = -5
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "individual"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SAlaF\_0.4_102"
props.path_to_experiment = rf"F:\ssNMR\20260113_SAlaF_0.4M_Ala"
props.procno = 102
props.expno = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
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
    63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    73,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.start_analysis()

props = settings.Properties()
props.prefit = True
props.spectrum_for_prefit = -1
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_0.4_103"
props.path_to_experiment = rf"F:\ssNMR\20260113_SAlaF_0.4M_Ala"
props.procno = 103
props.expno = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
props.subspec = [0, 100]


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
    63,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    73,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.start_analysis()
#############################################
props = settings.Properties()
props.prefit = False
props.spectrum_for_prefit = -5
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "individual"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_0.3_102"
props.path_to_experiment = rf"F:\ssNMR\20260113_SAlaF_0.3M_Ala"
props.procno = 102
props.expno = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
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
    63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    73,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.start_analysis()

props = settings.Properties()
props.prefit = True
props.spectrum_for_prefit = -1
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_0.3_103"
props.path_to_experiment = rf"F:\ssNMR\20260113_SAlaF_0.3M_Ala"
props.procno = 103
props.expno = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
props.subspec = [0, 100]


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
    63,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    73,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.start_analysis()

#############################################
props = settings.Properties()
props.prefit = False
props.spectrum_for_prefit = -5
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "individual"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_0.2_102"
props.path_to_experiment = rf"F:\ssNMR\20260113_SAlaF_0.2M_Ala"
props.procno = 102
props.expno = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
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
    63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    73,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.start_analysis()

props = settings.Properties()
props.prefit = True
props.spectrum_for_prefit = -1
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_0.2_103"
props.path_to_experiment = rf"F:\ssNMR\20260113_SAlaF_0.2M_Ala"
props.procno = 103
props.expno = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
props.subspec = [0, 100]


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
    63,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    73,
    peak_sign="-",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.start_analysis()

#############################################
props = settings.Properties()
props.prefit = False
props.spectrum_for_prefit = -5
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "individual"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_0.1_102"
props.path_to_experiment = rf"F:\ssNMR\20260113_SAlaF_0.1M_Ala"
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
    63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    73,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.start_analysis()

props = settings.Properties()
props.prefit = True
props.spectrum_for_prefit = -1
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_0.1_103"
props.path_to_experiment = rf"F:\ssNMR\20260113_SAlaF_0.1M_Ala"
props.procno = 103
props.expno = [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
props.subspec = [0, 100]


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
    62,
    peak_sign="+",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
# ds.add_peak(73, peak_sign="-",line_broadening={"gamma":{"max":2},"sigma":{"max":2}})
ds.start_analysis()

#############################################
props = settings.Properties()
props.prefit = False
props.spectrum_for_prefit = -5
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "individual"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_0.05_102"
props.path_to_experiment = rf"F:\ssNMR\20260112_SAlaF_0.05M_Ala"
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
    63,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.add_peak(
    73,
    peak_sign="+",
    line_broadening={"gamma": {"max": 4}, "sigma": {"max": 4}},
)
ds.start_analysis()

props = settings.Properties()
props.prefit = True
props.spectrum_for_prefit = -1
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = rf"C:\Users\Florian Taube\Desktop\SAlaF\Ala_0.05_103"
props.path_to_experiment = rf"F:\ssNMR\20260112_SAlaF_0.05M_Ala"
props.procno = 103
props.expno = [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
props.subspec = [0, 100]


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
    62,
    peak_sign="+",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.add_peak(
    73,
    peak_sign="+",
    line_broadening={"gamma": {"max": 2}, "sigma": {"max": 2}},
)
ds.start_analysis()
