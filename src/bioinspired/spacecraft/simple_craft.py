"""Simple spacecraft class for bioinspired design.
This module contains a simple spacecraft class that inherits from the base spacecraft class.
It provides a basic implementation of a spacecraft design with a name and a method to get its model.
"""

try:
    from .spacecraft_base import SpacecraftBase
except ImportError:
    from spacecraft_base import SpacecraftBase
    
class SimpleCraft(SpacecraftBase):
    """Simple spacecraft class.
    
    This class provides a basic implementation of a spacecraft design with a name and a method to get its model.
    It inherits from the base spacecraft class and implements the required methods.
    """

    def __init__(self, body_model):
        """Initialize the simple spacecraft with a name."""
        self._mass = 1000.0 # Spacecraft mass [kg]
        print("Name of the spacecraft:", self.get_name())
        super().__init__("SimpleCraft". body_model)
    
    def insert_into_body_model(self, body_model):
        """Self-inserts the spacecraft into a provided body model."""
        super().insert_into_body_model(body_model)
        print("Setting spacecraft mass to:", self._mass, "kg")
        body_model.get(self.name).mass = self._mass
    
    
def main():
    """Main function to demonstrate the SimpleCraft class."""
    simple_craft = SimpleCraft()
    print("SimpleCraft instance created with name:", simple_craft.get_name())
    
if __name__ == "__main__":
    main()