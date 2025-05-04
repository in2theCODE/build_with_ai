import importlib
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import yaml

from src.services.shared.constants.models import VerificationResult
from src.services.shared.constants.enums import ComponentType







class SynthesisSystem:
    """Core system for program synthesis with statistical verification."""

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the synthesis system.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path) if isinstance(config_path, str) else config_path
        self.logger = self._setup_logger()
        self.config = self._load_config()

        # Initialize core services
        self.spec_parser = self._initialize_component(ComponentType.SPECIFICATION_PARSER)
        self.synthesis_engine = self._initialize_component(ComponentType.SYNTHESIS_ENGINE)
        self.statistical_verifier = self._initialize_component(ComponentType.STATISTICAL_VERIFIER)
        self.code_generator = self._initialize_component(ComponentType.CODE_GENERATOR)
        self.feedback_collector = self._initialize_component(ComponentType.FEEDBACK_COLLECTOR)
        self.knowledge_base = self._initialize_component(ComponentType.KNOWLEDGE_BASE)

        # Initialize optimization services
        self.incremental_synthesis = self._initialize_component(ComponentType.INCREMENTAL_SYNTHESIS)
        self.verification_stratifier = self._initialize_component(ComponentType.VERIFICATION_STRATIFIER)
        self.language_interop = self._initialize_component(ComponentType.LANGUAGE_INTEROP)
        self.meta_learner = self._initialize_component(ComponentType.META_LEARNER)
        self.constraint_relaxer = self._initialize_component(ComponentType.CONSTRAINT_RELAXER)
        self.spec_inference = self._initialize_component(ComponentType.SPEC_INFERENCE)
        self.version_manager = self._initialize_component(ComponentType.VERSION_MANAGER)
        self.distributed_verifier = self._initialize_component(ComponentType.DISTRIBUTED_VERIFIER)
        self.symbolic_executor = self._initialize_component(ComponentType.SYMBOLIC_EXECUTOR)
        self.interface_contractor = self._initialize_component(ComponentType.INTERFACE_CONTRACTOR)

        self.logger.info("Synthesis system initialized")

    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the synthesis system."""
        logging_config = {
            'level': logging.INFO,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }

        logging.basicConfig(**logging_config)
        return logging.getLogger('synthesis-system')

    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_component(self, component_type: ComponentType) -> Any:
        """Initialize a system component based on configuration."""
        component_config = self.config.get('services', {}).get(component_type.value, {})
        component_class = component_config.get('class')

        # Check if this component is optional
        optional_components = [
            ComponentType.META_LEARNER.value,
            ComponentType.SYMBOLIC_EXECUTOR.value,
            ComponentType.DISTRIBUTED_VERIFIER.value,
            ComponentType.INCREMENTAL_SYNTHESIS.value,
            ComponentType.VERIFICATION_STRATIFIER.value,
            ComponentType.LANGUAGE_INTEROP.value,
            ComponentType.CONSTRAINT_RELAXER.value,
            ComponentType.SPEC_INFERENCE.value,
            ComponentType.VERSION_MANAGER.value,
            ComponentType.INTERFACE_CONTRACTOR.value
        ]

        # If this component is optional and not configured, return None
        if not component_class and component_type.value in optional_components:
            self.logger.info(f"Optional component {component_type.value} not configured, skipping")
            return None

        # For required services, raise an error if not configured
        required_components = [
            ComponentType.SPECIFICATION_PARSER.value,
            ComponentType.SYNTHESIS_ENGINE.value,
            ComponentType.STATISTICAL_VERIFIER.value,
            ComponentType.CODE_GENERATOR.value,
            ComponentType.FEEDBACK_COLLECTOR.value,
            ComponentType.KNOWLEDGE_BASE.value
        ]

        if not component_class and component_type.value in required_components:
            raise ValueError(f"No class specified for required component: {component_type.value}")

        # If we reach here and have no class, just return None
        if not component_class:
            return None

        # Dynamic import of the component class
        try:
            module_path, class_name = component_class.rsplit('.', 1)
            module = importlib.import_module(module_path)
            component_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to import component {component_class}: {e}")
            raise ImportError(f"Could not import {component_class}: {e}")

        # Initialize with configuration
        try:
            return component_cls(**component_config.get('params', {}))
        except Exception as e:
            self.logger.error(f"Failed to initialize component {component_class}: {e}")
            raise ValueError(f"Could not initialize {component_class}: {e}")

    def generate_from_spec(self,
                           specification: str,
                           context: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate code from a specification with verification.

        Args:
            specification: The formal specification or requirements
            context: Additional context for the synthesis

        Returns:
            Tuple of (generated_code, metadata)
        """
        self.logger.info(f"Starting code generation from specification")
        start_time = time.time()
        context = context or {}

        # Try to infer more details from the specification if the component exists
        if self.spec_inference:
            inferred_spec_data = self.spec_inference.enhance_specification(specification, context)
            if inferred_spec_data:
                specification = inferred_spec_data.get('enhanced_spec', specification)
                context = {**context, **inferred_spec_data.get('inferred_context', {})}
                self.logger.info("Enhanced specification with inferred details")

        # Check version manager for prior versions if component exists
        prior_versions = []
        if self.version_manager:
            prior_versions = self.version_manager.find_prior_versions(specification, context)
            if prior_versions:
                self.logger.info(f"Found {len(prior_versions)} prior versions of this specification")

        # Check knowledge base for cached results
        cache_key = self._compute_cache_key(specification, context)
        cached_result = self.knowledge_base.get(cache_key)
        if cached_result:
            self.logger.info(f"Found cached result for specification")
            # Update metadata with version info if it exists
            if self.version_manager:
                self.version_manager.record_usage(cache_key, cached_result)
            return cached_result['code'], cached_result['metadata']

        # Parse the specification into a formal model
        try:
            formal_spec = self.spec_parser.parse(specification, context)
        except Exception as e:
            self.logger.error(f"Failed to parse specification: {e}")
            raise ValueError(f"Could not parse specification: {e}")

        # Determine if we should use incremental synthesis
        use_incremental = False
        incremental_components = []
        if self.incremental_synthesis and formal_spec.is_decomposable():
            use_incremental = True
            incremental_components = self.incremental_synthesis.decompose(formal_spec)
            self.logger.info(f"Using incremental synthesis with {len(incremental_components)} services")

        # Perform synthesis
        try:
            if use_incremental:
                # Synthesize incrementally
                component_results = []
                for component_spec in incremental_components:
                    component_result = self.synthesis_engine.synthesize(component_spec)
                    component_results.append(component_result)

                # Combine the services
                synthesis_result = self.incremental_synthesis.combine(component_results)
            else:
                # Standard synthesis
                synthesis_result = self.synthesis_engine.synthesize(formal_spec)
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            raise ValueError(f"Synthesis failed: {e}")

        # Verify the result
        try:
            # Use verification stratification if available
            if self.verification_stratifier:
                verification_result = self.verification_stratifier.stratified_verify(
                    synthesis_result,
                    formal_spec
                )
            else:
                # Traditional verification
                verification_result = self.statistical_verifier.verify(
                    synthesis_result,
                    formal_spec
                )
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            raise ValueError(f"Verification failed: {e}")

        # Add symbolic execution tests if component exists
        if self.symbolic_executor and verification_result.status == VerificationResult.VERIFIED:
            try:
                symbolic_tests = self.symbolic_executor.generate_tests(synthesis_result, formal_spec)
                if symbolic_tests:
                    symbolic_result = self.symbolic_executor.execute_tests(symbolic_tests, synthesis_result)
                    if not symbolic_result.passed:
                        verification_result.status = VerificationResult.COUNTEREXAMPLE_FOUND
                        verification_result.reason = "Failed symbolic execution tests"
                        verification_result.counterexamples.extend(symbolic_result.failing_tests)
            except Exception as e:
                self.logger.warning(f"Symbolic execution failed: {e}")
                # Continue even if symbolic execution fails

        # If verification fails but we have constraint relaxation, try relaxing constraints
        if verification_result.status != VerificationResult.VERIFIED and self.constraint_relaxer:
            self.logger.info("Attempting constraint relaxation after verification failure")
            try:
                relaxed_spec = self.constraint_relaxer.relax_constraints(formal_spec, verification_result)
                if relaxed_spec:
                    self.logger.info("Successfully relaxed constraints, retrying synthesis")
                    relaxed_synthesis = self.synthesis_engine.synthesize(relaxed_spec)
                    relaxed_verification = self.statistical_verifier.verify(relaxed_synthesis, relaxed_spec)

                    if relaxed_verification.status == VerificationResult.VERIFIED:
                        self.logger.info("Verification succeeded with relaxed constraints")
                        synthesis_result = relaxed_synthesis
                        verification_result = relaxed_verification
                        formal_spec = relaxed_spec
                        synthesis_result.used_relaxation = True
            except Exception as e:
                self.logger.warning(f"Constraint relaxation failed: {e}")
                # Continue even if relaxation fails

        # Generate the final code
        try:
            if verification_result.status == VerificationResult.VERIFIED:
                if self.language_interop:
                    target_language = context.get('target_language', self.config.get('system', {}).get('default_language', 'python'))
                    generated_code = self.language_interop.generate_for_language(
                        synthesis_result,
                        target_language
                    )
                else:
                    generated_code = self.code_generator.generate(synthesis_result)

                # Verify interface contracts if component exists
                interface_valid = True
                if self.interface_contractor:
                    try:
                        interface_result = self.interface_contractor.verify_interfaces(
                            generated_code,
                            context.get('interface_contracts', {})
                        )
                        interface_valid = interface_result.is_valid

                        if not interface_valid:
                            self.logger.warning(f"Generated code fails interface contracts: {interface_result.failures}")
                    except Exception as e:
                        self.logger.warning(f"Interface verification failed: {e}")
                        # Continue even if interface verification fails

                # Only store fully valid code in the knowledge base
                if interface_valid:
                    # Store in knowledge base
                    metadata = {
                        'synthesis_time': synthesis_result.time_taken,
                        'verification_time': verification_result.time_taken,
                        'confidence': verification_result.confidence,
                        'used_incremental': use_incremental,
                        'used_relaxation': getattr(synthesis_result, 'used_relaxation', False),
                        'total_time': time.time() - start_time
                    }

                    # Store the result
                    self.knowledge_base.store(cache_key, {
                        'code': generated_code,
                        'metadata': metadata
                    })

                    # Record in version manager if it exists
                    if self.version_manager:
                        self.version_manager.record_new_version(cache_key, specification, context, metadata)

                    # Update meta-learner with successful strategy if component exists
                    if self.meta_learner:
                        self.meta_learner.record_success(
                            specification,
                            context,
                            getattr(synthesis_result, 'strategy', 'default')
                        )

                    return generated_code, metadata
                else:
                    # Interface contract failure
                    raise ValueError(f"Generated code does not satisfy interface contracts")
            else:
                # Handle verification failure
                self.logger.error(f"Verification failed: {verification_result.reason}")
                self.feedback_collector.record_failure(
                    specification,
                    context,
                    synthesis_result,
                    verification_result
                )

                # Update meta-learner with failed strategy if component exists
                if self.meta_learner:
                    self.meta_learner.record_failure(
                        specification,
                        context,
                        getattr(synthesis_result, 'strategy', 'default')
                    )

                # Try to repair or return best effort solution
                if self.config.get('system', {}).get('allow_best_effort', False) or context.get('allow_best_effort', False):
                    best_effort_code = self.code_generator.generate_best_effort(synthesis_result)
                    metadata = {
                        'best_effort': True,
                        'verification_status': verification_result.status.value,
                        'verification_reason': verification_result.reason,
                        'total_time': time.time() - start_time
                    }
                    return best_effort_code, metadata
                else:
                    raise ValueError(f"Failed to generate verified code: {verification_result.reason}")
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            raise ValueError(f"Code generation failed: {e}")

    def _compute_cache_key(self, specification: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Compute a cache key for the knowledge base."""
        context_str = json.dumps(context or {}, sort_keys=True)
        return hashlib.sha256(f"{specification}:{context_str}".encode()).hexdigest()
