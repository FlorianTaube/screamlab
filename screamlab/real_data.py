from screamlab import settings, dataset
import sys

props = settings.Properties()
props.prefit = True
props.spectrum_for_prefit = -1
props.buildup_types = ["exponential", "biexponential"]
props.spectrum_fit_type = "global"
props.output_folder = (
    rf"C:\Users\Florian Taube\Desktop\Prolin_auswertung\P-100K_N"
)

props.path_to_experiment = (
    rf"F:\NMR\Max\20230620_100mM_H2N-PG-OH_10mM_AMUPOL_1p3mm_18kHz_DNP_100K"
)
props.procno = 103
props.expno = [54, 59]

ds = dataset.Dataset()
ds.props = props
ds.add_peak(
    50,
    peak_sign="+",
    line_broadening={"sigma": {"max": 3}, "gamma": {"max": 3}},
    peak_label="ProN",
)
ds.start_analysis()


sys.exit()

liste = ["1_50", "51_100", "101_150", "151_200", "201_250"]

for i in liste:
    props = settings.Properties()
    props.prefit = False
    props.spectrum_for_prefit = -1
    props.buildup_types = ["exponential", "biexponential"]
    props.spectrum_fit_type = "numint"
    props.output_folder = rf"C:\Users\Florian Taube\Desktop\dasgrauen\{i}"

    props.path_to_experiment = rf"F:\ssNMR\20260211_SCREAM_ILT_13C3_Ala_{i}"
    props.procno = 103
    props.expno = [1, 50]

    ds = dataset.Dataset()
    ds.props = props
    ds.add_peak(-6, peak_sign="+", integration_range=[39, 55])
    ds.start_analysis()
