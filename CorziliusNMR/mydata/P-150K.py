from CorziliusNMR import dataset, settings
import sys

props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Pro-150K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(175, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(46, peak_label="Cdelta")
ds.add_peak(29, peak_label="Cbeta")
ds.add_peak(25, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()
