"""JSON spacecraft base class for BioInspired spacecraft simulation.
This module provides a base class for spacecraft designs that can be serialized to and from JSON.
It will load a JSON configuration file based on the spacecraft name (self.name), and look for it in the folder containing the spacecraft design.
"""

import json
import numpy as np
from abc import abstractmethod

from .spacecraft_base import SpacecraftBase


class JSONSpacecraftBase(SpacecraftBase):
    """Base class for spacecraft designs that can be serialized to and from JSON.
    This class extends the SpacecraftBase class to include methods for JSON serialization.
    Each spacecraft design should inherit from this class and implement the required methods.
    """

    def __init__(self, **kwargs):
        """Initialize the JSON spacecraft with a name and initial state."""
        super().__init__(**kwargs)
        self._load_config()

    @abstractmethod
    def required_properties(self) -> dict[str, list[str]]:
        """Return a list of required properties for the spacecraft configuration.
        Example format:
        {
            Engine: [position, direction, max_thrust],
            RigidBodyProperties: [dry_mass, fuel_mass, inertia_tensor],
        }
        """
        raise NotImplementedError(
            "Subclasses must implement required_properties method."
        )

    def _load_config(self) -> dict:
        """Load spacecraft configuration from JSON file based on the spacecraft name.
        It then adds all properties to the spacecraft object.
        The JSON file should be located in the same directory as this module.
        """
        config_path = f"src/bioinspired/spacecraft/{self.name}.json"
        try:
            with open(config_path, "r") as file:
                config = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {config_path}: {e}")
        self._apply_config(config)
        return config

    def _validate_config(self, config: dict) -> None:
        """Validate the loaded configuration against required properties.
        Checks recursively if sub-properties are present somewhere in the property value.
        :param config: The loaded configuration dictionary.
        :return: None
        :raises ValueError: If required properties are missing or incorrectly formatted.
        """

        def contains_sub_prop(value, sub_prop):
            """Recursively check if sub_prop is present in value."""
            if isinstance(value, dict):
                if sub_prop in value:
                    return True
                return any(contains_sub_prop(v, sub_prop) for v in value.values())
            elif isinstance(value, list):
                return any(contains_sub_prop(item, sub_prop) for item in value)
            return False

        required_props = self.required_properties()
        for prop, sub_props in required_props.items():
            if prop not in config:
                raise ValueError(f"Missing required property: {prop}")
            for sub_prop in sub_props:
                if not contains_sub_prop(config[prop], sub_prop):
                    raise ValueError(
                        f"Missing required sub-property '{sub_prop}' in '{prop}'."
                    )

    def _apply_config(self, config: dict):
        """Apply the loaded configuration to the spacecraft object."""
        self._validate_config(config)
        # Set properties based on the configuration
        for prop, value in config.items():
            # if not hasattr(self, prop):
            if isinstance(value, list):
                self.__dict__["_" + prop] = np.array(value)
            else:
                self.__dict__["_" + prop] = value
            # else:
            #     raise UserWarning(
            #         f"Property {prop} is already defined in the spacecraft class and is overwritting the JSON configuration."
            #     )

