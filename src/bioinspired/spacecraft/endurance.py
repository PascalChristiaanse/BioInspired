"""Endurance spacecraft module for BioInspired project.
This module defines the Endurance spacecraft class, which inherits from the SimpleCraft class.
"""

import numpy as np
from overrides import override

from tudatpy.numerical_simulation.propagation_setup import torque, acceleration

from .spacecraft_base import SpacecraftBase
from .JSON_spacecraft_base import JSONSpacecraftBase
from .rotating_spacecraft_base import RotatingSpacecraftBase


class Endurance(RotatingSpacecraftBase, JSONSpacecraftBase):
    """Endurance spacecraft class.

    This class represents the Endurance spacecraft, inheriting from SpacecraftBase and JSONSpacecraftBase.
    It initializes the spacecraft with specific parameters and provides methods for spacecraft operations.
    """

    def __init__(
        self, simulation, initial_state=None, name: str = "Endurance", **kwargs
    ):
        super().__init__(
            name=name,
            simulation=simulation,
            initial_state=initial_state[:6]
            if initial_state is not None
            else np.array([0, 0, 0, 0, 0, 0]),
            initial_rotational_state=initial_state[6:]
            if initial_state is not None
            else np.array([1, 0, 0, 0, 0, 0, 7.121]),  # 68 RPM TARS estimate
            **kwargs,
        )

    @property
    @override
    def mass(self) -> float:
        """Return the mass of the Endurance spacecraft."""
        return self._mass  # pylint: disable=no-member

    @property
    @override
    def inertia_tensor(self) -> np.ndarray:
        """Return the inertia tensor of the Endurance spacecraft."""
        return self._inertia_tensor  # pylint: disable=no-member

    @property
    @override
    def center_of_mass(self) -> np.ndarray:
        """Return the center of mass of the Endurance spacecraft."""
        return self._center_of_mass  # pylint: disable=no-member

    @override
    def _get_torque_settings(self) -> dict[str, dict[str, list[torque.TorqueSettings]]]:
        return {self.name: {self.name: []}}

    @override
    def _get_acceleration_settings(
        self,
    ) -> dict[str, dict[str, list[acceleration.AccelerationSettings]]]:
        return super()._get_acceleration_settings()

    @override
    def required_properties(self) -> dict[str, list[str]]:
        return {
            "mass": [],
            "inertia_tensor": [],
            "center_of_mass": [],
        }
