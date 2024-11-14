"""
io module of the CorziliusNMR package.
"""

import sys
import os
from datetime import date
from bruker.data.nmr  import *
from matplotlib import pyplot as plt
import numpy as np
import bruker.api.topspin as top
def generate_csv_from_scream_set(path, output_file, expno, procnos=[103]):
### Path to the Topspin files
    expno = np.arange(expno[0], expno[1] + 1)
    for procno in procnos:
        ### Name of the output file
        par_out = np.zeros(shape=(4,len(expno)))
        import bruker.api.topspin as top
        top = top.Topspin()
        dp = top.getDataProvider()
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