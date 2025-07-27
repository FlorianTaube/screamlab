============
screamlab
============

Welcome! This is screamlab, a Python package for the evaluation of relaxation processes in multi-spin system observed by  SCREAM-DNP (Specific Cross Relaxation Enhancement by Active Motions under Dynamic Nuclear Polarization).


Features
========

A list of features:

* Spectral deconvolution with various possible lineshapes (Gaussian, Lorentzian, Voigtian)

* Global fitting of spectra with consistent parameters

* Subsequent fitting of peak areas over time with exponential or biexponential behavior, or involving Solomon equations for a more complex fitting.


And to make it even more convenient for users and future-proof:

* Open source project written in Python (>= 3.7)

* High test coverage

* Extensive user and API documentation




.. warning::
    screamlab is currently under active development and still considered in Beta development state. Therefore, expect frequent changes in features and public APIs that may break your own code. Nevertheless, feedback as well as feature requests are highly welcome.


Installation
============

To install the screamlab package on your computer (sensibly within a Python virtual environment), open a terminal (activate your virtual environment), and type in the following:

.. code-block:: bash

    pip install screamlab


License
=======

This program is free software: you can redistribute it and/or modify it under the terms of the **BSD License**.



.. toctree::
   :maxdepth: 2
   :caption: User Manual:
   :hidden:

   audience
   usecases
   installing

.. toctree::
   :maxdepth: 2
   :caption: Developers:
   :hidden:

   people
   deploying
   developers
   changelog
   roadmap
   api/modules

