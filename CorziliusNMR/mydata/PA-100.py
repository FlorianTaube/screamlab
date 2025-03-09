from CorziliusNMR import dataset, settings
import sys

props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)

props.path_to_experiment = (
    r"F:\NMR\Max\20230817_100mM_HN-PA-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)

props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\PA-100K"
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
        "sigma": {"min": 1, "max": 1.4},
        "gamma": {"min": 0.4, "max": 0.8},
    },
)
ds.add_peak(
    26,
    peak_label="Cgamma",
    line_broadening={
        "sigma": {"min": 1.73, "max": 2.13},
        "gamma": {"min": 0.0, "max": 0.84},
    },
)
ds.start_buildup_fit_from_topspin()
