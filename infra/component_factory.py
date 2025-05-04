"""Component factory for the program synthesis system.

This module provides factory functions to create and configure services
for the program synthesis system, with a focus on neural techniques.
"""

import logging
import importlib
from typing import Dict, Any, Optional, Type

from src.services.shared.models.base import BaseComponent
from src.services.shared.models.enums import Components as ComponentType
from src.services.neural_code_generator.app.enhanced_neural_code_generator import EnhancedNeuralCodeGenerator


class ComponentFactory:
    """Factory for creating and configuring system services."""

    def __init__(self):
        """Initialize the component factory."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registered_services = {}
        self._register_default_services()

    def _register_default_services(self):
        """Register default component implementations."""
        # Register the neural code generator
        self.register_component(
            ComponentType.CODE_GENERATOR,
            "neural_code_generator",
            EnhancedNeuralCodeGenerator
        )

    def register_component(self, component_type: ComponentType, name: str, component_class: Type[BaseComponent]):
        """
        Register a component implementation.

        Args:
            component_type: The type of component
            name: A unique name for this implementation
            component_class: The component class
        """
        key = f"{component_type.value}.{name}"
        self.registered_services[key] = component_class
        self.logger.debug(f"Registered component: {key}")

    def create_component(self, component_type: ComponentType, implementation: str,
                         params: Dict[str, Any]) -> Optional[BaseComponent]:
        """
        Create a component of the specified type with the given implementation.

        Args:
            component_type: The type of component to create
            implementation: The name of the implementation to use
            params: Parameters for component initialization

        Returns:
            The created component or None if creation fails
        """
        key = f"{component_type.value}.{implementation}"

        # Check if the component is registered
        if key in self.registered_services:
            component_class = self.registered_services[key]
            try:
                component = component_class(**params)
                self.logger.info(f"Created component {key}")
                return component
            except Exception as e:
                self.logger.error(f"Failed to create component {key}: {e}")
                return None

        # Try to dynamically import the component
        try:
            # Construct the expected module path
            module_name = f"program_synthesis_system.services"
            if component_type.value:
                module_name += f".{component_type.value}"
            module_name += f".{implementation}"

            # Try to import the module
            module = importlib.import_module(module_name)

            # Get the component class (assume it's the first class that inherits from BaseComponent)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseComponent) and attr != BaseComponent:
                    try:
                        component = attr(**params)
                        # Register for future use
                        self.register_component(component_type, implementation, attr)
                        self.logger.info(f"Dynamically created component {key}")
                        return component
                    except Exception as e:
                        self.logger.error(f"Failed to instantiate component {key}: {e}")
                        return None

            self.logger.error(f"No suitable component class found in module {module_name}")
            return None

        except ImportError as e:
            self.logger.error(f"Failed to import component {key}: {e}")
            return None

    def create_from_config(self, config: Dict[str, Any]) -> Dict[str, BaseComponent]:
        """
        Create services from a configuration dictionary.

        Args:
            config: Configuration dictionary with component specifications

        Returns:
            Dictionary mapping component types to created services
        """
        services = {}

        if "services" not in config:
            self.logger.warning("No services section in configuration")
            return services

        for component_type, component_config in config["services"].items():
            try:
                # Get the component class and parameters
                if isinstance(component_config, dict):
                    class_path = component_config.get("class", "")
                    params = component_config.get("params", {})

                    # Extract implementation name from class path
                    # e.g., "program_synthesis_system.services.ast_code_generator.neural_code_generator.NeuralCodeGenerator"
                    # would have implementation "neural_code_generator"
                    parts = class_path.split(".")
                    if len(parts) >= 3:
                        implementation = parts[-2]  # Second to last part
                    else:
                        implementation = parts[-1].lower()  # Fallback to lowercase class name

                    # Try to create the component
                    try:
                        component_enum = ComponentType(component_type)
                    except ValueError:
                        self.logger.warning(f"Unknown component type: {component_type}")
                        continue

                    component = self.create_component(component_enum, implementation, params)
                    if component:
                        services[component_type] = component

            except Exception as e:
                self.logger.error(f"Failed to create component {component_type}: {e}")

        return services