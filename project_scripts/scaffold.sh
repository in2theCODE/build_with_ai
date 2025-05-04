#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error

# Directory to scaffold into (default is current directory)
TARGET_DIR="${1:-.}"

# Check if target directory exists, create if it doesn't
if [ ! -d "$TARGET_DIR" ]; then
  mkdir -p "$TARGET_DIR"
fi

# Change to the target directory
cd "$TARGET_DIR"

# Function to create directories and handle errors
create_dirs() {
  if ! mkdir -p "$@" 2>/dev/null; then
    echo "Error: Failed to create directories: $@" >&2
    return 1
  fi
}

echo "🚀 Starting directory structure scaffolding..."
echo "📂 Creating directory structure in: $(pwd)"

# Create main structure
echo "📁 Creating base structure..."
create_dirs src/templates/domains/{web,mobile,desktop}
create_dirs src/templates/domains/game/{2d,3d,mobile,console}

# AI/ML structure
echo "📁 Creating AI/ML structure..."
create_dirs src/templates/domains/ai_ml/model_development/{experiment_tracking,feature_store,model_versioning,hyperparameter_optimization,distributed_training,transfer_learning,automl,data_augmentation,model_documentation,neural_architecture_search}
create_dirs src/templates/domains/ai_ml/model_deployment/{model_serving,container_deployment,model_conversion,multi_model_serving,resource_optimization,dark_launch,canary_deployment,model_registry,edge_deployment,model_compression}
create_dirs src/templates/domains/ai_ml/monitoring_feedback/{performance_monitoring,drift_detection,ab_testing,explainability,feature_importance,feedback_loop,adversarial_testing,metrics_collection,automatic_retraining,model_deprecation}

# Data domain
create_dirs src/templates/domains/data

# IoT and Edge Computing
echo "📁 Creating IoT and Edge Computing structure..."
create_dirs src/templates/domains/iot/edge_computing/edge_to_cloud/{device_registration,data_synchronization,edge_gateway,edge_function_deployment,cloud_fallback,edge_message_queue,device_shadow,edge_state_management,data_prioritization,edge_security}
create_dirs src/templates/domains/iot/edge_computing/offline_first/{offline_storage,conflict_resolution,offline_auth,background_sync,progressive_enhancement,action_queue,workflow_engine,offline_analytics,data_compression,sync_status}
create_dirs src/templates/domains/iot/edge_computing/low_latency_processing/{real_time_processing,stream_processing,local_inference,time_series_processing,low_latency_database,memory_optimized_compute,predictive_caching,data_locality,resource_aware_scheduling,event_driven_processing}

# Blockchain, AR/VR, Embedded, Cloud
create_dirs src/templates/domains/{blockchain,ar_vr,embedded,cloud}

# DevOps and DevSecOps
echo "📁 Creating DevOps and DevSecOps structure..."
create_dirs src/templates/domains/devops/devsecops/security_scanning/{sast,dast,container_scanning,dependency_scanning,secret_detection,iac_security,api_security,fuzzing,license_compliance,code_review}
create_dirs src/templates/domains/devops/devsecops/compliance_as_code/{control_mapping,compliance_reporting,policy_as_code,compliance_testing,compliance_monitoring,compliance_dashboard,audit_trail,evidence_collection,regulatory_change,compliance_documentation}
create_dirs src/templates/domains/devops/devsecops/vulnerability_management/{vulnerability_workflow,patch_automation,cve_monitoring,remediation_prioritization,risk_assessment,zero_day_response,scanning_schedule,fix_verification,advisory_distribution,post_incident_review}

# Security
create_dirs src/templates/domains/security

# Cross Platform
echo "📁 Creating Cross Platform structure..."
create_dirs src/templates/domains/cross_platform/code_sharing/{shared_business_logic,state_management,platform_agnostic_models,isomorphic_javascript,kotlin_multiplatform,flutter_widgets,react_native_components,xamarin_forms,dart_domain,cpp_core_library}
create_dirs src/templates/domains/cross_platform/api_design/{universal_api,graphql_schema,api_versioning,platform_adapters,offline_first_api,cross_platform_auth,rest_mapping,rate_limiting,error_handling,capability_detection}
create_dirs src/templates/domains/cross_platform/user_experience/{responsive_design,adaptive_ui,design_tokens,theming_engine,accessibility_components,multi_layout_engine,animation_system,interaction_patterns,device_adaptation,common_navigation}

# Real-time Systems
echo "📁 Creating Real-time Systems structure..."
create_dirs src/templates/domains/real_time_systems/websocket_applications/{websocket_server,data_synchronization,connection_management,session_persistence,notification_service,presence_awareness,activity_feed,websocket_auth,rate_limiting,connection_recovery}
create_dirs src/templates/domains/real_time_systems/event_streaming/{kafka_integration,stream_processing,stream_analytics,stream_aggregation,windowed_processing,stream_joins,stream_filtering,time_based_processing,event_replay,stream_connector}
create_dirs src/templates/domains/real_time_systems/low_latency_data/{memory_optimized_structures,zero_copy_transfer,single_writer_principle,thread_affinity,lock_free_algorithms,disruptor_pattern,data_locality,cache_aware_algorithms,memory_pooling,data_prefetching}

# Internationalization
echo "📁 Creating Internationalization structure..."
create_dirs src/templates/domains/internationalization/multi_language/{translation_management,string_externalization,message_bundles,language_detection,dynamic_translation,plural_forms,gender_text,contextual_translation,translation_memory,machine_translation}
create_dirs src/templates/domains/internationalization/rtl_support/{rtl_layout,bidirectional_text,rtl_components,mirroring_service,rtl_navigation,direction_detection,rtl_animation,bidi_algorithm,mixed_direction_handler,image_flipping}
create_dirs src/templates/domains/internationalization/cultural_adaptation/{date_time_format,number_format,currency_format,address_format,cultural_preference,regional_content,locale_sorting,unit_conversion,color_significance,regulatory_compliance}

# Industry
echo "📁 Creating Industry structure..."
create_dirs src/templates/domains/industry/ecommerce
create_dirs src/templates/domains/industry/finance/pci_dss/{self_assessment,network_segmentation,tokenization,logging_monitoring,vulnerability_management,secure_code_review,access_control,data_encryption,payment_gateway,penetration_testing}
create_dirs src/templates/domains/industry/healthcare/hipaa/{risk_assessment,data_flow_mapping,api_authorization,data_encryption,audit_logging,business_associate,breach_notification,deidentification,consent_management,data_retention}
create_dirs src/templates/domains/industry/{education,social,entertainment,enterprise}
create_dirs src/templates/domains/industry/data_privacy/gdpr_ccpa/{dsar_handler,impact_assessment,processing_agreement,cookie_consent,privacy_policy,right_to_be_forgotten,data_portability,legitimate_interest,cross_border_transfer,activity_register}

# Languages
echo "📁 Creating Languages structure..."
create_dirs src/templates/languages/{python,javascript,typescript,java,csharp,cpp,go,rust,swift,kotlin,php,ruby}

# Frameworks
create_dirs src/templates/frameworks/{backend,frontend,mobile,game,ai_ml,data}

# Components
echo "📁 Creating Components structure..."
create_dirs src/templates/services/{auth,database,api}
create_dirs src/templates/services/ui/{forms,tables,navigation,charts,layouts}
create_dirs src/templates/services/ui/accessibility/wcag_compliance/{compliance_audit,component_library,focus_management,keyboard_navigation,skip_navigation,color_contrast,alt_text,form_validation,screen_reader_announcements,modal_dialog}
create_dirs src/templates/services/ui/accessibility/assistive_technology/{screen_reader_components,voice_navigation,switch_control,alternative_input,magnification,voice_over,high_contrast,reduced_motion,read_aloud,caption_management}
create_dirs src/templates/services/ui/accessibility/testing_validation/{automated_testing,accessibility_linting,unit_test_framework,user_flow_validation,mobile_testing,screen_reader_testing,regression_testing,documentation_generator,user_testing_scripts,issue_prioritization}
create_dirs src/templates/services/{payment,search,caching,messaging,storage,logging,testing}

# Patterns
echo "📁 Creating Patterns structure..."
create_dirs src/templates/patterns/architectural/{mvc,mvvm,microservices,clean_architecture}
create_dirs src/templates/patterns/architectural/serverless/event_driven/{event_bridge,schema_registry,event_sourcing,event_store,serverless_processor,pub_sub,event_replay,async_handling,event_versioning,idempotent_consumer}
create_dirs src/templates/patterns/architectural/serverless/function_composition/{function_chaining,orchestrator,fan_out_fan_in,step_function,saga_pattern,routing_framework,composition_gateway,aggregator_function,api_composition,circuit_breaker}
create_dirs src/templates/patterns/architectural/serverless/optimization/{cold_start,warm_up,memory_optimization,dependency_management,connection_pool,function_decomposition,parameter_optimization,context_reuse,parallel_processing,timeout_optimization}
create_dirs src/templates/patterns/{creational,structural,behavioral}
create_dirs src/templates/patterns/sustainability/energy_efficient_code/{low_power_design,battery_aware,energy_profiling,efficient_algorithms,optimized_data_structures,power_management,sleep_mode,cpu_gpu_optimization,memory_optimization,io_batching}
create_dirs src/templates/patterns/sustainability/carbon_aware_deployment/{deployment_scheduler,green_region_selection,carbon_monitoring,carbon_based_scaling,workload_deferral,carbon_calculator,offset_integration,renewable_usage,efficiency_reporting,carbon_aware_cicd}
create_dirs src/templates/patterns/sustainability/resource_optimization/{caching_strategies,resource_allocation,optimized_containers,memory_usage,query_optimization,asset_compression,infrastructure_sizing,cache_optimization,content_delivery,resource_cleanup}

# Infrastructure
create_dirs src/templates/infrastructure/{docker,kubernetes,terraform,aws,azure,gcp,ci_cd}

# Project Types
create_dirs src/templates/project_types/{mvp,prototype,enterprise,saas}

# Template Engine
create_dirs src/template_engine/{parsers,generators,validators,helpers}

# Adapters
create_dirs src/adapters/{vcs,ai,testing,deployment}

# Config
create_dirs src/config/{linting,formatting,build,package_managers}

# Metadata
create_dirs src/metadata/{dependencies,compatibility,tags}

# Create empty files in each directory to ensure the structure is maintained
echo "📄 Creating placeholder files in all directories..."
FOLDER_COUNT=$(find src -type d | wc -l)
find src -type d | while read dir; do
  touch "$dir/.gitkeep"
done

# Create a README.md in the root directory
cat > README.md << 'EOF'
# Project Structure

This repository contains a comprehensive project structure for various domains and technologies.

## Directory Structure

The project follows a hierarchical structure organized by domains, languages, frameworks, components, patterns, and more.

## Usage

Navigate through the directories to find templates and code samples for your specific use case.

## License

This project structure is provided as-is. Feel free to use it for your projects.
EOF

# Count the total number of directories created
total_dirs=$(find src -type d | wc -l)
echo "✅ Directory structure scaffolding complete!"
echo "📊 Statistics:"
echo "  - Total directories created: $total_dirs"
echo "  - All directories have placeholder .gitkeep files"
echo "  - README.md created in the root directory"
echo ""
echo "🎉 Success! Your project structure is ready to use."