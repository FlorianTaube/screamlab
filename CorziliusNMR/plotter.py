import matplotlib.pyplot as plt


class Plotter():

    def __init__(self,dataset):
        self._dataset = dataset
        self._experiments = dataset.experiments
        self._output_file_info = \
            dataset._fileNames.generate_export_output_pdf_file_name()
        print(self._output_file_info)
        print("hello")

    def plot(self):
        pass

class ExperimentalSpectraPlotter(Plotter):

    def __init__(self,dataset):
        super().__init__(dataset)

    def plot(self):
        for experiment in self._experiments:
            plt.plot(experiment.x_axis,experiment.y_axis,label=f"tdel = {experiment.tbup} s")
        plt.xlabel("$chemical\ shift$ / ppm")
        plt.ylabel("$signal\ intensity$ / a.u.")
        plt.title("First view of exported data")
        plt.legend()
        plt.savefig(self._output_file_info,format="pdf")
        plt.close
