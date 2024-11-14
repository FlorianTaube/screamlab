from CorziliusNMR import io,fitting


path = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
expno=[24,33]
output_file = r"C:\Users\Florian Taube\Desktop\Prolin_auswertung_Test\HN-P-100K"
io.generate_csv_from_scream_set(path, output_file, expno)
peak_list = [172,60,50,40,30]
plus_minus_list = ["-"]*5
buildup_fit_type_list = ["Biexponential_with_offset","Biexponential"]
fitting.scream_buildup_time_evaluation(output_file,peak_list,
                                       plus_minus_list=plus_minus_list,
                                       autopeakpick=True,
                                       buildup_fit_type_list=buildup_fit_type_list)
