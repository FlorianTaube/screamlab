from CorziliusNMR import dataset, settings
import sys

props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)

props.path_to_experiment = r"F:\ssNMR\20250121_S48F_125K"

props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\AP-125K"
)

ds = dataset.Dataset()
ds.props = props
ds.add_peak(178, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(48, peak_label="Cdelta")
ds.add_peak(30, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()
