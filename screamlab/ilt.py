class InverseLaplaceTransformation:
    def __init__(self, spectra):
        self.spectra = spectra

    def start_ilt(self):
        # import iltpy
        import iltpy as ilt

        # other libraries for handling data, plotting
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        print(f"ILTpy version: {ilt.__version__}")

        # Load data using numpy arrays
        result_list = []
        result_tau = None
        print(len(self.spectra[0].y_axis))
        plt.plot(self.spectra[0].x_axis, self.spectra[-1].y_axis)
        plt.gca().invert_xaxis()
        plt.show()
        plt.close()
        for ppm_nr, ppm in enumerate(self.spectra[0].x_axis):
            print(rf"{ppm_nr+1} / {len(self.spectra[0].x_axis)}")
            t_EPR = []
            data_EPR = []
            for i in self.spectra:
                t_EPR.append(i.tpol)
            for nr, i in enumerate(self.spectra):
                data_EPR.append(i.y_axis[ppm_nr])

            data_EPR = np.array(data_EPR)
            t_EPR = np.array(t_EPR)

            t = t_EPR
            y = data_EPR

            tau, A_tau = self.inverse_laplace_nnls(t, y, n_tau=205)
            result_list.append(list(A_tau))
            from scipy.signal import find_peaks

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata

        x = self.spectra[0].x_axis
        y = tau
        hoehen = np.array(result_list)

        import matplotlib.pyplot as plt
        import numpy as np

        # generate 2 2d grids for the x & y bounds
        y, x = np.meshgrid(tau, self.spectra[0].x_axis)

        z = result_list
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        fig, ax = plt.subplots()

        # c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        c = ax.pcolormesh(x, y, z, cmap="RdBu", vmin=-0.01e6, vmax=0.01e6)
        ax.set_title("pcolormesh")
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        ax.invert_xaxis()
        ax.set_yscale("log")
        fig.colorbar(c, ax=ax)

        plt.show()
        import sys

        sys.exit()

    def inverse_laplace_nnls(self, t, y, n_tau=410):
        """
        Inverse Laplace Transformation mit Non-Negative Least Squares.

        t : Array mit Zeitpunkten (>0)
        y : Messdaten
        n_tau : Anzahl diskrete Relaxationszeiten
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.optimize import nnls
        from scipy.signal import find_peaks

        t_nonzero = t[t > 0]
        tau = np.logspace(
            np.log10(min(t_nonzero)), np.log10(max(t_nonzero)), n_tau
        )

        K = 1 - np.exp(-np.outer(t, 1 / tau))

        a, _ = nnls(K, y)

        return tau, a
