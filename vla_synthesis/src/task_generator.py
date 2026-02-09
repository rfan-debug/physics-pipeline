import numpy as np
import random

# Try importing genesis, if not available, it will be mocked in tests or fail if not installed in production
try:
    import genesis as gs
except ImportError:
    gs = None

class TaskGenerator:
    def __init__(self):
        self.current_object = None
        self.instruction = ""
        self.target_object_entity = None

        # Define ASSET_DB with lambdas to create primitives
        # Using lambdas allows deferred creation and parameterization
        # We assume gs.morphs has Box, Sphere, Cylinder and they accept pos and color
        if gs:
            self.ASSET_DB = {
                "cube": lambda pos, color: gs.morphs.Box(pos=pos, size=(0.04, 0.04, 0.04), color=color),
                "sphere": lambda pos, color: gs.morphs.Sphere(pos=pos, radius=0.03, color=color),
                "mug": lambda pos, color: gs.morphs.Cylinder(pos=pos, height=0.08, radius=0.04, color=color) # Placeholder
            }
        else:
            self.ASSET_DB = {}

        self.COLOR_DB = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0)
        }

    def reset_task(self, scene):
        """
        Reset the task: clear previous objects, spawn a new random object, and generate instruction.

        Args:
            scene: The genesis scene object.
        """
        if gs is None:
             raise ImportError("Genesis library is not installed. Please install it to use TaskGenerator.")

        # Clear previous objects if supported
        # We try to remove the entity if the scene supports it
        if self.target_object_entity is not None:
            if hasattr(scene, 'remove_entity'):
                try:
                    scene.remove_entity(self.target_object_entity)
                except Exception as e:
                     print(f"Warning: Failed to remove previous entity: {e}")
            else:
                 # If remove_entity is not available, we just log a warning or do nothing
                 pass

        # Reset state
        self.target_object_entity = None
        self.instruction = ""

        # Randomly select object and color
        obj_name = random.choice(list(self.ASSET_DB.keys()))
        color_name = random.choice(list(self.COLOR_DB.keys()))
        color_rgb = self.COLOR_DB[color_name]

        # Randomize position
        # x in [0.3, 0.7], y in [-0.2, 0.2]
        # z should be slightly above table to avoid collision/penetration
        x = random.uniform(0.3, 0.7)
        y = random.uniform(-0.2, 0.2)
        z = 0.05
        pos = (x, y, z)

        # Create the object primitive
        # Retrieve the lambda and call it
        morph_creator = self.ASSET_DB[obj_name]
        try:
             morph = morph_creator(pos, color_rgb)
        except Exception as e:
             # Fallback or error handling
             raise RuntimeError(f"Failed to create morph for {obj_name}: {e}")

        # Add entity to scene
        self.target_object_entity = scene.add_entity(morph)

        # Generate instruction
        # Templates: "Pick up the {color} {object}", "Grasp the {color} item", "Move the {object}"
        templates = [
            "Pick up the {color} {object}",
            "Grasp the {color} item",
            "Move the {object}",
            "Retrieve the {color} {object}"
        ]
        template = random.choice(templates)

        # Format the instruction
        try:
            # Handle cases where {color} might be missing in template (e.g. "Move the {object}")
            self.instruction = template.format(color=color_name, object=obj_name)
        except KeyError:
            # Should not happen with current templates but good for safety
            self.instruction = template.replace("{object}", obj_name).replace("{color}", color_name)

    def get_instruction(self):
        """
        Get the current task instruction and target object.

        Returns:
            tuple: (instruction_string, target_object_entity)
        """
        return self.instruction, self.target_object_entity
