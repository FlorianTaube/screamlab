from typing import Any


class Properties:
    def __init__(
        self,
        prefit: bool = False,
        buildup_types: list = ["exponential"],
        spectrum_fit_type: list = ["global"],
        spectrum_for_prefit: int = 0,
        plot_prefit=True,
    ):
        self._prefit = None
        self.prefit = prefit
        self._buildup_types = None
        self.buildup_types = buildup_types
        self._spectrum_for_prefit = None
        self.spectrum_for_prefit = spectrum_for_prefit
        self._spectrum_fit_type = None
        self.spectrum_fit_type = spectrum_fit_type

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
    def spectrum_for_prefit(self) -> list:
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
