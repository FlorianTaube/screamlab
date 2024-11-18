"""
io module of the CorziliusNMR package.
"""
import CorziliusNMR.dataset
import numpy as np
from bruker.data.nmr import *
import bruker.api.topspin as top
import os

'''
def generate_csv_from_scream_set(path, output_file, expno, procnos=[103]):
### Path to the Topspin files
    expno = np.arange(expno[0], expno[1] + 1)
    for procno in procnos:
        ### Name of the output file
        par_out = np.zeros(shape=(4,len(expno)))

        counter = 1
        for exp in expno:
            try:
                PROTON = sys.argv[1]
            except:
                #PROTON = top.getInstallationDirectory() + '\examdata'
                p3  = r"pdata"
                PROTON = os.path.join(path, str(exp), p3, str(procno))
                #PROTON = p1 + p2

            proton = dp.getNMRData(PROTON)
            if proton == None:
                raise Exception('Dataset {} does not exist'.format(PROTON))
            ns = float(proton.getPar("NS"))
            phc0 = float(proton.getPar("PHC0"))
            phc1 = float(proton.getPar("PHC1"))
            par_out[0,counter-1] = ns
            par_out[1,counter-1] = phc0
            par_out[2,counter-1] = phc1

            l20 = float(proton.getPar("L 20"))
            tbup = l20/4

            par_out[3,counter-1] = tbup

            specData = proton.getSpecDataPoints()
            pr = specData['physicalRanges'][0]
            left = float(pr['start'])
            right = float(pr['end'])

            ### Frequency domain points
            axis = np.linspace(left,right,len(specData['dataPoints']))
            if counter == 1:
                npout = axis
                header = "# 13C chemical shift / ppm,"
            else:
                npout  = np.vstack((npout,axis))
            ### Intensity domain points
            ydata = np.divide(specData['dataPoints'],ns)
            #ydata = ydata + counter*-1e5*numpy.ones_like(specData['dataPoints'])
            npout = np.vstack((npout,ydata))
            header += "Normalized Intensity after " + str(tbup) + " s / arb. units,"
            plt.plot(axis,ydata,label=exp)
            plt.legend()
            plt.xlabel("chemical shift / ppm")
            plt.ylabel("Intensity normalized to number of scans / a.u.")

            counter += 1

        today = str(date.today())
        npout = np.transpose(npout)
        comment = "#Date of generation: " + today + "\n"
        comment += "#Each experiment was divided by the applied number of scans to retrieve the normalized intensity after the corresponding buildup time. \n"
        comment += "#Number of points: " + str(len(ydata)) + "\n"
        comment += "#Number of scans:\n#" + np.array2string(par_out[0,:], max_line_width=100, precision=2,separator=",") + "\n"
        comment += "#Phase correction 0th order / degree:\n#" + np.array2string(par_out[1,:], max_line_width=100, precision=2,separator=",") + "\n"
        comment += "#Phase correction 1st order / degree:\n#" + np.array2string(par_out[2,:], max_line_width=100, precision=2,separator=",") + "\n"
        np.set_printoptions(suppress=True)
        comment += "#Delay time for build up / s:\n#" + np.array2string(par_out[3,:], max_line_width=120, precision=2,separator=",") + "\n"
        plt.savefig(output_file+"_"+str(procno)+".pdf", dpi='figure',
                                        format="pdf")
        np.savetxt((output_file+"_"+str(procno)+".csv"),npout,header=header,
                   delimiter=",",comments = comment, fmt="%10.5f")
        plt.close()
'''



class TopspinImporter:

    def __init__(self,dataset):
        self._dataset = dataset
        self._top = top.Topspin()
        self._data_provider = self._top.getDataProvider()
        self._current_path_to_exp = None
        self._nmr_data = None

    def import_topspin_data(self):
        pass





class ScreamImporter(TopspinImporter):

    def __init__(self,dataset):
        super().__init__(dataset)

    def import_topspin_data(self):
        path = self._dataset.file_name_generator.path_to_topspin_experiment
        procno =self._dataset.file_name_generator.procno_of_topspin_experiment
        for expno_nr,expno in \
                enumerate(self._dataset.file_name_generator.expno_of_topspin_experiment):
            self._dataset.experiments.append(CorziliusNMR.dataset.Experiment())
            self._current_path_to_exp = r"F:\NMR\Max\20230706_100mM_HN-P-OH_10mM_AUPOL_1p3mm_18kHz_DNP_100K"
            self._nmr_data = self._data_provider.getNMRData(
                os.path.join(path, str(expno), "pdata", str(procno)))
            print(os.path.join(path, str(expno), "pdata", str(procno)))
            print(self._nmr_data)

            #self._dataset._set_values()

    def _set_values(self):
        self._get_number_of_scans()
        self._get_buildup_time()
        self._get_x_axis()
        self._get_y_axis()
        self._normalize_y_values_to_number_of_scans()

    #def _set_nmr_data():


    #def _get_number_of_scans(self):
    #    self.NS = int(self._nmr_metadata.getPar("NS"))

    '''
    def _get_buildup_time(self):
        self.tbup = int(self._nmr_metadata.getPar("L 20")) / 4

    def _get_x_axis(self):
        _physicalRange = self._nmr_spectral_data['physicalRanges'][0]
        _number_of_datapoints = self._nmr_spectral_data['dataPoints']
        self.x_axis = np.linspace(float(_physicalRange['start']),
                                  float(_physicalRange[ 'end']),
                                  len(_number_of_datapoints))

    def _get_y_axis(self):
        self.y_axis = self._nmr_spectral_data['dataPoints']

    def _normalize_y_values_to_number_of_scans(self):
        self.y_axis = np.divide(self.y_axis, self.NS)
'''








class Pseudo2DImporter(TopspinImporter):

    def __init__(self,dataset):
        super().__init__(dataset)

    def import_topspin_data(self):
        return

    def read_in_topspin_data():
        pass










