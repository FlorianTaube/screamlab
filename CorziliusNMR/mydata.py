from CorziliusNMR import dataset, settings
import sys

props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
    spectrum_for_prefit=-2,
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
ds.add_peak(32, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()

sys.exit()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230818_100mM_HN-PA-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\PA-150K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(171, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(48, peak_label="Cdelta")
ds.add_peak(31, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


sys.exit()


sys.exit()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential", "exponential"],
    expno=[24, 32],
    subspec=[200, 0],
    spectrum_for_prefit=-4,
)
props.path_to_experiment = r"F:\ssNMR\20241128_S36F_100K"
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Pro-na-100K"
)
ds = dataset.Dataset()
ds.props = props
# ds.add_peak(175, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(46, peak_label="Cdelta")
ds.add_peak(30, peak_label="Cbeta")
ds.add_peak(24, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential", "exponential"],
    expno=[24, 32],
    subspec=[200, 0],
    spectrum_for_prefit=-2,
)
props.path_to_experiment = r"F:\ssNMR\20241203_S36F_150K"
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Pro-na-150K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(176, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(45, peak_label="Cdelta")
ds.add_peak(30, peak_label="Cbeta")
ds.add_peak(24, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()
sys.exit()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential", "exponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = r"F:\ssNMR\20250120_S47F_100K"
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\GP-100K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(176, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(49, peak_label="Cdelta")
ds.add_peak(30, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = r"F:\ssNMR\20250120_S47F_125K"
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\GP-125K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(176, peak_label="CO")
ds.add_peak(72, peak_label="Glycerol_1")

ds.add_peak(61, peak_label="Calpha")
ds.add_peak(49, peak_label="Cdelta")
ds.add_peak(30, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = r"F:\ssNMR\20250120_S47F_150K"
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\GP-150K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(176, peak_label="CO")
ds.add_peak(72, peak_label="Glycerol_1")
ds.add_peak(63, peak_label="Glycerol_2")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(49, peak_label="Cdelta")
ds.add_peak(30, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


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


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = r"F:\ssNMR\20250121_S48F_150K"
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\AP-150K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(178, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(48, peak_label="Cdelta")
ds.add_peak(30, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = r"F:\ssNMR\20250224_S49F_100K"
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\AcPro-na-100K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(178, peak_label="CO")
ds.add_peak(174, peak_label="Acetyl")
ds.add_peak(73, peak_label="Glycerol2")
ds.add_peak(63, peak_label="Glycerol1")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(50, peak_label="Cdelta")
ds.add_peak(32, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.add_peak(23, peak_label="CH3")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = r"F:\ssNMR\20250224_S49F_125K"
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\AcPro-na-125K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(178, peak_label="CO")
ds.add_peak(174, peak_label="Acetyl")
ds.add_peak(73, peak_label="Glycerol2")
ds.add_peak(63, peak_label="Glycerol1")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(50, peak_label="Cdelta")
ds.add_peak(32, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.add_peak(23, peak_label="CH3")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = r"F:\ssNMR\20250224_S49F_150K"
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\AcPro-na-150K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(178, peak_label="CO")
ds.add_peak(174, peak_label="Acetyl")
ds.add_peak(73, peak_label="Glycerol2")
ds.add_peak(63, peak_label="Glycerol1")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(50, peak_label="Cdelta")
ds.add_peak(32, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.add_peak(23, peak_label="CH3")
ds.start_buildup_fit_from_topspin()


sys.exit()


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
ds.add_peak(31, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230818_100mM_HN-PA-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\PA-150K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(171, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(48, peak_label="Cdelta")
ds.add_peak(31, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20231218_100mM_Ac_N-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\AcPro-100K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(178, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(50, peak_label="Cdelta")
ds.add_peak(32, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20231218_100mM_Ac_N-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_125K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\AcPro-125K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(178, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(50, peak_label="Cdelta")
ds.add_peak(32, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 25, 26, 27, 28, 29, 31, 32],
    subspec=[200, 0],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20231220_100mM_Ac_N-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_150K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\AcPro-150K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(178, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(50, peak_label="Cdelta")
ds.add_peak(32, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()

props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Pro-100K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(175, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(46, peak_label="Cdelta")
ds.add_peak(29, peak_label="Cbeta")
ds.add_peak(25, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_125K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Pro-125K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(175, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(46, peak_label="Cdelta")
ds.add_peak(29, peak_label="Cbeta")
ds.add_peak(25, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


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


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230620_100mM_H2N-PG-OH_10mM_AMUPOL_1p3mm_18kHz_DNP_100K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\PG-100K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(172, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(48, peak_label="Cdelta")
ds.add_peak(32, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = (
    r"F:\NMR\Max\20230815_1M_HN-P-OH_na_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
)
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\Pro-na-100K"
)
ds = dataset.Dataset()
ds.props = props
# ds.add_peak(172, peak_label="CO")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(46, peak_label="Cdelta")
ds.add_peak(30, peak_label="Cbeta")
ds.add_peak(24, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()


props = settings.Properties(
    prefit=True,
    buildup_types=["biexponential"],
    expno=[24, 32],
    subspec=[200, 0],
)
props.path_to_experiment = r"F:\ssNMR\20250120_S47F_125K"
props.output_folder = (
    r"C:\Users\Florian Taube\Desktop\Prolin_auswertung\GP-125K"
)
ds = dataset.Dataset()
ds.props = props
ds.add_peak(176, peak_label="CO")
ds.add_peak(72, peak_label="Glycerol_1")
ds.add_peak(63, peak_label="Glycerol_2")
ds.add_peak(61, peak_label="Calpha")
ds.add_peak(49, peak_label="Cdelta")
ds.add_peak(30, peak_label="Cbeta")
ds.add_peak(26, peak_label="Cgamma")
ds.start_buildup_fit_from_topspin()
