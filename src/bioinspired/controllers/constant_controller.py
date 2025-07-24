"""Constant Controller Module
This module provides a constant controller for spacecraft, which applies a fixed control action for debugging purposes.
"""

from overrides import override
import numpy as np

from bioinspired.controllers.controller_base import ControllerBase


class ConstantController(ControllerBase):
    """Constant controller for spacecraft.
    This controller applies a fixed control action for debugging purposes.
    """

    def __init__(
        self,
        simulator,
        lander_name: str,
        target_name: str = None,
        control_vector: np.ndarray = None,
    ):
        """Initialize the constant controller."""
        super().__init__(simulator, lander_name, target_name)
        if control_vector is not None:
            self._control_vector = control_vector
        else:
            self._control_vector = np.zeros(24)
            self._control_vector[13] = 1
            self._control_vector[8] = 1
            self._control_vector[4] = 0.1
            # self._control_vector[12] = 1


    @override
    def get_control_action(self) -> np.ndarray:
        """Return a constant control action."""
        return self._control_vector
