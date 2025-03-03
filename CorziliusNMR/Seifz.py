from CorziliusNMR import dataset, settings
import sys


props = settings.Properties(
    prefit=True, expno=[2], procno=1, subspec=[-25, -40]
)
props.path_to_experiment = r"F:\ssNMR\20250220_AG05_01"
props.output_folder = r"C:\Users\Florian Taube\Desktop\Seifz\AG-05"
ds = dataset.Dataset()
ds.props = props
ds.add_peak(-33)
ds.start_buildup_fit_from_topspin()
