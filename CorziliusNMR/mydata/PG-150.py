from CorziliusNMR import dataset, settings
import sys

props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
    # spectrum_for_prefit=-3
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230814_100mM_H2N-PG-OH_10mM_AMUPOL_1p3mm_18kHz_DNP_150K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\PG-150K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(172, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(48, peak_label="Cdelta")
ds.add_peak(32, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()
