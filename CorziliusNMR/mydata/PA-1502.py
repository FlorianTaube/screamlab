from CorziliusNMR import dataset, settings
import sys


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20231026_100mM_HN-PA-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K_2"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\PA-150K_2"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(171, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(48, peak_label="Cdelta")
ds.add_peak(
    32,
    peak_label="Cbeta",
    line_broadening={
        "sigma": {"min": 0.87, "max": 1.27},
        "gamma": {"min": 0.87, "max": 1.27},
    },
)
ds.add_peak(
    26,
    peak_label="Cgamma",
    line_broadening={
        "sigma": {"min": 1.56, "max": 1.96},
        "gamma": {"min": 0.44, "max": 0.84},
    },
)
ds.start_buildup_fit_from_topspin()
