"""
The `settings` module is responsible for managing configuration options used
in the analysis process.
"""

from typing import Any
import os


class Properties:
    """
    A class to manage and validate properties related to spectral fitting and buildup types.

    Attributes
    ----------

    expno : list of int
        Experiment numbers. Default is None.
    prefit : bool, optional
        Indicates whether prefit mode is enabled. Default is False.
    spectrum_for_prefit : int, optional
        Specifies the spectrum index for prefit. Default is -1, meaning the last spectrum in
        :obj:`CorziliusNMR.settings.Spectra`.
    buildup_types : list of str, optional
        A list of buildup function types. Default is ["exponential"]. Options:
        "biexponential", "exponential_with_offset", "biexponential_with_offset",
        "stretched_exponential".
    spectrum_fit_type : str, optional
        A list specifying the spectrum fit type: "global" (default) or "independent".
    procno : int, optional
        Process number. Default is 103 (standard in SCREAM-DNP experiments).
    loop20 : str, optional
        Loop parameter. Default is "L 20" (standard in SCREAM-DNP experiments).
    delay20 : str, optional
        Delay parameter. Default is "D 20" (standard in SCREAM-DNP experiments).
    subspec : list, optional
        Two floats defining the spectrum range to cut. Must be within the valid spectrum range.
        Default is [].

    """

    def __init__(
        self,
        prefit: bool = False,
        buildup_types: list = None,
        spectrum_fit_type: list = None,
        spectrum_for_prefit: int = -1,
        path_to_experiment: str = "",
        procno: int = 103,
        expno: list = None,
        loop20: str = "L 20",
        delay20: str = "D 20",
        output_folder: str = "",
        subspec=[],
    ):
        self.init_call = True
        if buildup_types is None:
            buildup_types = ["exponential"]
        if spectrum_fit_type is None:
            spectrum_fit_type = ["global"]
        if expno is None:
            expno = [1]
        self._path_to_experiment = None
        self.path_to_experiment = path_to_experiment
        self._procno = None
        self.procno = procno
        self._expno = None
        self.expno = expno
        self._prefit = None
        self.prefit = prefit
        self._buildup_types = None
        self.buildup_types = buildup_types
        self._spectrum_for_prefit = None
        self.spectrum_for_prefit = spectrum_for_prefit
        self._spectrum_fit_type = None
        self.spectrum_fit_type = spectrum_fit_type
        self._loop20 = None
        self.loop20 = loop20
        self._delay20 = None
        self.delay20 = delay20

        self._output_folder = None
        self.output_folder = output_folder
        self.subspec = subspec
        self.init_call = False

    def __str__(self):
        return (
            f"[[Settings]]\n"
            f"Experiment folder: {self.path_to_experiment}\n"
            f"Expno: {self.expno}\n"
            f"Procno: {self.procno}\n"
            f"Prefit: {self.prefit}\n"
            f"Spectrum for prefit: {self.spectrum_for_prefit}\n"
            f"Spectrum fitting type: {self.spectrum_fit_type}\n"
            f"Buildup evaluation: {self.buildup_types}\n"
            f"Calculated polarization time from {self.loop20} and "
            f"{self.delay20} if SCREAM data given.\n"
            f"Wrote output files to: {self.output_folder}"
        )

    @property
    def output_folder(self) -> str:
        """
        Folder path for saving output. Default is an empty string ("").

        Returns
        -------
            str: The current folder path where the output is being saved.

        """
        return self._output_folder

    @output_folder.setter
    def output_folder(self, value: Any):
        """Set the folder path for saving output."""
        if not isinstance(value, str):
            raise TypeError(
                f"Expected 'output_folder' to be of type 'str', got {type(value).__name__}."
            )
        if not os.path.exists(f"{value}") and not self.init_call:
            os.makedirs(f"{value}")

        self._output_folder = value

    @property
    def delay20(self) -> str:
        return self._delay20

    @delay20.setter
    def delay20(self, value: Any):
        # TODO Write controls
        self._delay20 = value

    @property
    def loop20(self) -> str:
        return self._loop20

    @loop20.setter
    def loop20(self, value: Any):
        # TODO Write controls
        self._loop20 = value

    @property
    def expno(self) -> list:
        return self._expno

    @expno.setter
    def expno(self, value: Any):
        if not isinstance(value, list):
            raise TypeError(
                f"Expected 'expno' to be of type 'list', got {type(value).__name__}."
            )
        if not all(isinstance(item, int) for item in value):
            raise ValueError(
                "All elements in the 'expno' list must be of type 'int'."
            )
        if len(value) == 2:
            value = list(range(value[0], value[-1] + 1))
        self._expno = [str(item) for item in value]

    @property
    def procno(self) -> int:
        return self._procno

    @procno.setter
    def procno(self, value: Any):
        if not isinstance(value, int):
            raise TypeError(
                f"Expected 'procno' to be of type 'int', got {type(value).__name__}."
            )
        self._procno = str(value)

    @property
    def path_to_experiment(self) -> str:
        """
        str: Path to the experiment data.

        Returns the path where experiment-related data is stored.
        This is typically a directory path in string format.

        Default is an empty string ("").
        """
        return self._path_to_experiment

    @path_to_experiment.setter
    def path_to_experiment(self, value: Any):
        """
        Sets the path to the experiment data.

        Parameters
        ----------
        value : str
            The path to set. Must be a string.

        Raises
        ------
        TypeError
            If `value` is not a string.
        """
        if not isinstance(value, str):
            raise TypeError(
                f"Expected 'path_to_experiment' to be of type 'str', got {type(value).__name__}."
            )
        self._path_to_experiment = value

    @property
    def spectrum_fit_type(self) -> list:
        return self._spectrum_fit_type

    @spectrum_fit_type.setter
    def spectrum_fit_type(self, value: Any):
        allowed_values = {
            "global",
            "individual",
            "hight",
        }
        if not isinstance(value, list):
            raise TypeError(
                f"Expected 'spectrum_fit_type' to be of type 'list', got {type(value).__name__}."
            )
        if not all(item in allowed_values for item in value):
            raise ValueError(
                f"All elements in 'spectrum_fit_type' must be one of {allowed_values}."
            )
        if not value:
            raise ValueError("'spectrum_fit_type' cannot be an empty list.")
        self._spectrum_fit_type = value

    @property
    def spectrum_for_prefit(self) -> int:
        return self._spectrum_for_prefit

    @spectrum_for_prefit.setter
    def spectrum_for_prefit(self, value: Any):
        if not isinstance(value, int):
            raise TypeError(
                f"Expected 'spectrum_for_prefit' to be of type 'int', got {type(value).__name__}."
            )
        self._spectrum_for_prefit = value

    @property
    def buildup_types(self) -> list:
        return self._buildup_types

    @buildup_types.setter
    def buildup_types(self, value: Any):
        allowed_values = {
            "exponential",
            "biexponential",
            "biexponential_with_offset",
            "exponential_with_offset",
            "streched_exponential",
        }
        if not isinstance(value, list):
            raise TypeError(
                f"Expected 'buildup_types' to be of type 'list', got {type(value).__name__}."
            )
        if not all(item in allowed_values for item in value):
            raise ValueError(
                f"All elements in 'buildup_types' must be one of {allowed_values}."
            )
        if not value:
            raise ValueError("'buildup_types' cannot be an empty list.")
        self._buildup_types = value

    @property
    def prefit(self) -> bool:
        return self._prefit

    @prefit.setter
    def prefit(self, value: Any):
        if not isinstance(value, bool):
            raise TypeError(
                f"Expected 'prefit' to be of type 'bool', got {type(value).__name__}."
            )
        self._prefit = value
