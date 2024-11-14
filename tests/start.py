# Assuming `top` is a module you want to use
#import top  # Add this if top is a module you want to import

class TopspinExporter:
    def __init__(self, dataset):
        self._dataset = dataset

    def export(self):
        # Assuming top.Topspin() and top.getDataProvider() are valid calls.
        #topspin = top.Topspin()
        #dataprovider = top.getDataProvider()
        self.pathlist_to_experimental_data()  # Now this is valid
        pass

    # Add self as the first argument for instance methods
    def pathlist_to_experimental_data(self):
        # Implement your logic here
        pass

class ScreamExporter(TopspinExporter):
    def __init__(self, dataset):
        super().__init__(dataset)

    def export(self):
        print("hallo2")
        return

    # Add self as the first argument for instance methods
    def pathlist_to_experimental_data(self):
        print("hallo")
        pass

# Testing the implementation
dataset = "Some dataset"  # Placeholder for dataset
exporter = ScreamExporter(dataset)
exporter.export()  # This should print "hallo2" and return
