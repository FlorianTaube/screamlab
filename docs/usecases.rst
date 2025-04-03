.. _use_cases:

=========
Use cases
=========

This section provides several examples of how basic operations might be structured. A typical analysis routine is generally divided into two parts: first, the settings are defined using the :obj:`CorziliusNMR.settings.Properties` class, followed by the determination of peak information using :obj:`CorziliusNMR.datset.Dataset`.

Specifying properties
=====================
This Python code is configuring properties related to an experiment, specifically designed to handle SCREAM-DNP data provided from Bruker's TopSpin software. Important in any cases is to specify the path to a folder where the experiments are saved, determine the experiments to be analyzed (using experiment numbers), and defines the output folder for the results. The experiments can be provided in two different formats. A list like [1,8] means that all experiments from 1 to 8 should be included, or it can be more specific by defining a list with all the experiments, e.g., [1,5,9,12,13,18]."

.. code-block:: python
    :linenos:

    from CorziliusNMR import settings, dataset

    props = settings.Properties()
    props.path_to_experiment = r"/path/to/dataset"
    props.expno = [1, 8]
    props.output_folder = r"/path/to/output"

There is also the option of further, optional customisation depending on the requirements.  In many cases, it is practical to carry out a prefit on one specific spectrum in order to minimise computing times. To do this specify the following:

.. code-block:: python
    :linenos:

    props.prefit = True             # Default option is False
    props.spectrum_for_prefit = 3   # Default option is -1 meaning the last entry in the expno list

Note that the value in spectrum_for_prefit refers to the position on an experiment in the expno list and not to the experiment number.
Also it can be specified wich buildup model should be used for evaluation. There are four options: exponential, exponential with offset, biexponential, and biexponential with offset and either one or multiple can performt during one analysis by providing a list as follows:

.. code-block:: python
    :linenos:

    props.buildup_types = [ "exponential",]
    #Performs buidlup fit using a simple exponential. Default option

    props.buildup_types = [ "exponential", "exponential_with_offset", "biexponential", ]
    #Performs three buildup fits using first an exponential model,
    #then an exponential wiht offset followd by a biexponential model.

Adding peaks and start analysis
===============================
After specifying the properties, the dataset object must be instantiated and assigned to a variable.
Todo
.. code-block:: python
    :linenos:

    ds = dataset.Dataset()
    ds.props = props

    ds.add_peak(-16)
    ds.start_buildup_fit_from_topspin()
