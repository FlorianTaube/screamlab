from CorziliusNMR import dataset, settings
import sys


props = settings.Properties(buildup_types=["biexponential"], expno=[24, 32])

props.path_to_experiment = (
    r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
)

props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Pro-150K"
)

ds = dataset.Dataset()
ds.props = props
ds.add_peak(175, peak_label="CO", peak_sign="-", fitting_type="gauss")

ds = dataset.Dataset()
ds.props = props
ds.add_peak(61)
ds.add_peak(46, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(29, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(25, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True, buildup_types=["biexponential"], expno=[24, 32]
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230817_100mM_HN-PA-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\PA-100K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(171, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(61, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(48, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(32, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(25, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.start_buildup_fit_from_topspin()

props = settings.Properties(
    prefit=True, buildup_types=["biexponential"], expno=[24, 32]
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230818_100mM_HN-PA-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\PA-100K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(171, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(61, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(48, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(32, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(25, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.start_buildup_fit_from_topspin()

props = settings.Properties(
    prefit=True, buildup_types=["biexponential"], expno=[24, 32]
)
props.path_to_experiment = (
    r"F:\NMR\Max\20231026_100mM_HN-PA-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K_2"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\PA-100K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(171, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(61, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(48, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(32, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(25, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True, buildup_types=["biexponential"], expno=[24, 32]
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230620_100mM_H2N-PG-OH_10mM_AMUPOL_1p3mm_18kHz_DNP_100K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\PG-100K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(172, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(60, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(48, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(32, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(26, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.start_buildup_fit_from_topspin()

props = settings.Properties(
    prefit=True, buildup_types=["biexponential"], expno=[24, 32]
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230814_100mM_H2N-PG-OH_10mM_AMUPOL_1p3mm_18kHz_DNP_150K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\PG-150K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(172, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(60, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(48, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(32, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(26, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.start_buildup_fit_from_topspin()

props = settings.Properties(
    prefit=True, buildup_types=["biexponential"], expno=[24, 32]
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Pro-100K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(175, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(61, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(46, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(29, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(25, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.start_buildup_fit_from_topspin()

props = settings.Properties(
    prefit=True, buildup_types=["biexponential"], expno=[24, 32]
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_125K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Pro-125K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(175, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(61, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(46, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(29, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(25, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True, buildup_types=["biexponential"], expno=[24, 32]
)
props.path_to_experiment = (
    r"F:\NMR\Max\20231218_100mM_Ac_N-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\AcPro-100K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(178, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(61, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(50, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(32, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(26, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.start_buildup_fit_from_topspin()

props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 25, 26, 27, 28, 29, 31, 32],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20231220_100mM_Ac_N-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\AcPro-150K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(178, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(61, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(50, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(32, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(26, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.start_buildup_fit_from_topspin()

props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 25, 26, 27, 28, 29, 31, 32],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20231218_100mM_Ac_N-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_125K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\AcPro-150K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(178, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(61, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(50, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(32, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.add_peak(26, line_broadening={"sigma": {"max": 2}, "gamma": {"max": 2}})
ds.start_buildup_fit_from_topspin()
