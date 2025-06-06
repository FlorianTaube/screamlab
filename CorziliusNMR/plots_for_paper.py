from CorziliusNMR import settings, dataset
import sys

props = settings.Properties()
props.path_to_experiment = (
    r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
props.expno = [25, 33]
props.prefit = True
props.spectrum_for_prefit = -1
props.buildup_types = ["biexponential", "exponential"]
props.output_folder = r"C:\Users\Florian Taube\Desktop\Test\delta_DP"

ds = dataset.Dataset()
ds.props = props
ds.add_peak(175, peak_sign="+", fitting_type="voigt")
ds.add_peak(60, peak_sign="+", fitting_type="voigt")
ds.add_peak(46, peak_sign="+", fitting_type="voigt")
ds.add_peak(30, peak_sign="+")
ds.add_peak(25, peak_sign="+")
ds.start_analysis()
sys.exit()
props = settings.Properties()
props.path_to_experiment = (
    r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
props.expno = [22, 33]
props.prefit = True
props.procno = 102
props.spectrum_for_prefit = -1
props.buildup_types = ["exponential", "biexponential"]
props.output_folder = r"C:\Users\Florian Taube\Desktop\Test\DPsat"

ds = dataset.Dataset()
ds.props = props
ds.add_peak(173, peak_sign="+", fitting_type="voigt")
# ds.add_peak(58,peak_sign='+')
# ds.add_peak(44,peak_sign='+')
# ds.add_peak(27,peak_sign='+')
# ds.add_peak(22,peak_sign='+')
ds.start_analysis()


sys.exit()


delays = [
    "0.0",
    "0.5",
    "0.25",
    "0.75",
    "1.5",
    "1.25",
    "1.75",
    "2.0",
    "2.5",
    "3.0",
    "3.5",
    "4.0",
    "8.0",
]


def gen_dataset(props):
    ds = dataset.Dataset()
    ds.props = props
    ds.add_peak(-6, peak_sign="+", line_broadening={"gamma": {"max": 4}})
    ds.add_peak(
        40,
        peak_sign="+",
    )
    ds.add_peak(
        49,
        peak_sign="+",
    )
    ds.start_analysis()


def gen_dataset_delta(props):
    ds = dataset.Dataset()
    ds.props = props
    ds.add_peak(-6, peak_sign="-", line_broadening={"gamma": {"max": 4}})
    ds.start_analysis()


def gen_dataset_long(props):
    ds = dataset.Dataset()
    ds.props = props
    ds.add_peak(-6, peak_sign="+", line_broadening={"gamma": {"max": 4}})
    ds.add_peak(
        40,
        peak_sign="+",
    )
    ds.add_peak(
        49,
        peak_sign="+",
    )
    ds.add_peak(
        -40,
        peak_sign="+",
    )
    ds.start_analysis()


def gen_dataset_S46F(props):
    ds = dataset.Dataset()
    ds.props = props
    ds.add_peak(-6, peak_sign="+", line_broadening={"gamma": {"max": 4}})
    ds.start_analysis()


for delay in delays:

    props = settings.Properties()
    props.path_to_experiment = rf"F:\ssNMR\20250110_46F_{delay}"
    props.buildup_types = [
        "exponential",
    ]
    props.spectrum_fit_type = "global"
    props.prefit = True
    props.procno = 102

    props.expno = [1, 18]
    props.output_folder = (
        rf"C:\Users\Florian Taube\Desktop\S46F\DPsat\dpsat_{delay}"
    )
    # gen_dataset_S46F(props)

    props.expno = [1, 5]
    props.output_folder = (
        rf"C:\Users\Florian Taube\Desktop\S46F\DPsat\dpsat_{delay}_1"
    )
    # gen_dataset(props)

    props.expno = [6, 9]
    props.output_folder = (
        rf"C:\Users\Florian Taube\Desktop\S46F\DPsat\dpsat_{delay}_2"
    )
    # gen_dataset(props)

    props.expno = [10, 13]
    props.output_folder = (
        rf"C:\Users\Florian Taube\Desktop\S46F\DPsat\dpsat_{delay}_3"
    )
    # gen_dataset_long(props)

    props.expno = [14, 18]
    props.spectrum_for_prefit = -3
    props.output_folder = (
        rf"C:\Users\Florian Taube\Desktop\S46F\DPsat\dpsat_{delay}_4"
    )
    # gen_dataset_long(props)

    props.buildup_types = [
        "exponential",
        "exponential_with_offset",
        "biexponential",
        "biexponential_with_offset",
    ]
    props.spectrum_fit_type = "global"
    props.prefit = True
    props.procno = 103

    props.expno = [1, 8]
    props.output_folder = (
        rf"C:\Users\Florian Taube\Desktop\S46F\deltaDPsat\dpsat_{delay}"
    )
    gen_dataset_delta(props)
    sys.exit()


sys.exit()


props = settings.Properties()
props.path_to_experiment = r"F:\ssNMR\20250110_45F_0.0"
props.spectrum_fit_type = "individual"
props.expno = [14, 18]
props.procno = 102
props.prefit = True
props.buildup_types = [
    "exponential",
    "biexponential",
    "biexponential_with_offset",
]
props.output_folder = r"C:\Users\Florian Taube\Desktop\T\dpsat0.0_4"
# .subspec = [13,-38]
ds = dataset.Dataset()
ds.props = props
ds.add_peak(-6, peak_sign="+", line_broadening={"gamma": {"max": 4}})
ds.add_peak(
    40,
    peak_sign="+",
)
ds.add_peak(
    49,
    peak_sign="+",
)
ds.add_peak(
    -30.8,
    peak_sign="+",
)
# ds.add_peak(119,peak_sign='+',)
ds.add_peak(
    -40,
    peak_sign="+",
)

ds.start_analysis()

sys.exit()
props = settings.Properties()
props.path_to_experiment = r"F:\ssNMR\20250422_S1"
props.expno = [20, 25]
props.prefit = True
props.spectrum_for_prefit = -3
props.buildup_types = ["exponential"]
props.output_folder = r"C:\Users\Florian Taube\Desktop\Test"
ds = dataset.Dataset()
ds.props = props


ds.add_peak(
    -13,
    peak_sign="+",
    fitting_type="voigt",
    line_broadening={"sigma": {"max": 60}, "gamma": {"max": 60}},
)
ds.add_peak(
    252,
    peak_sign="+",
    fitting_type="voigt",
    line_broadening={"sigma": {"max": 60}, "gamma": {"max": 60}},
)
ds.add_peak(
    115,
    peak_sign="+",
    fitting_type="voigt",
    line_broadening={"sigma": {"max": 60}, "gamma": {"max": 60}},
)
ds.add_peak(
    -144,
    peak_sign="+",
    fitting_type="voigt",
    line_broadening={"sigma": {"max": 60}, "gamma": {"max": 60}},
)
ds.add_peak(
    -284,
    peak_sign="+",
    fitting_type="voigt",
    line_broadening={"sigma": {"max": 60}, "gamma": {"max": 60}},
)
ds.start_analysis()

sys.exit()
