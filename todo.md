# RAGnRoll Framework - Configuration-Based Approach Implementation Plan

## Overview
This document outlines the plan for transitioning the RAGnRoll framework from a CLI-based approach to a configuration-file-based approach. The goal is to replace the current CLI with a system that is started with `python main.py` and automatically looks for a `config.py` file containing all settings and parameters.

## Current Architecture Analysis
The current system uses a CLI with Typer that supports multiple commands:
- `run_evaluations`: Main evaluation command with many parameters
- `split_data`: Data splitting functionality
- `test_generalization_error`: Testing generalization error
- `draw_pipeline`: Pipeline visualization

The system currently uses command-line arguments for all parameters and YAML files for pipeline configurations.

## Proposed New Architecture
The new system will:
1. Be started with `python main.py`
2. Automatically look for a `config.py` file
3. Store all settings and parameters in the configuration file
4. Support multiple configuration profiles that can be executed sequentially

## Implementation Plan

### Phase 1: Create the New Configuration System
1. Create a `main.py` file that will be the new entry point
2. Create a `config.py` file that will contain all configuration parameters
3. Implement a configuration parser that can handle multiple profiles
4. Create a mapping between CLI parameters and configuration parameters

### Phase 2: Migrate CLI Functionality
1. Migrate the `run_evaluations` command functionality to use the configuration file
2. Migrate the `split_data` command functionality
3. Migrate the `test_generalization_error` command functionality
4. Migrate the `draw_pipeline` command functionality

### Phase 3: Testing and Validation
1. Create a minimal test configuration file
2. Ensure the program is executable at each step
3. Validate that all functionality is preserved
4. Create documentation for the new configuration approach

### Phase 4: Documentation and Cleanup
1. Remove the old CLI code
2. Update documentation to reflect the new approach
3. Create examples of configuration files

## Detailed Implementation Steps

### Step 1: Create the main.py entry point
- This will be the new entry point for the application
- It will load the configuration file and execute the appropriate functions

### Step 2: Design the config.py structure
- Define a structure that can contain all the parameters currently passed via CLI
- Support multiple configuration profiles in a list format
- Include all necessary parameters for data paths, experiment settings, etc.

### Step 3: Implement configuration loading
- Create a function to load and validate the configuration file
- Implement profile selection and iteration logic

### Step 4: Migrate run_evaluations functionality
- Adapt the existing `run_evaluations` function to use configuration parameters instead of CLI arguments
- Ensure all current functionality is preserved

### Step 5: Migrate other CLI commands
- Adapt `split_data`, `test_generalization_error`, and `draw_pipeline` functions to use configuration parameters

### Step 6: Create test configuration
- Create a minimal `config.py` file that can be used for testing
- Ensure it works with a small dataset from start to finish

### Step 7: Validate and test
- Test the new system with the minimal configuration
- Ensure all functionality works as expected
- Verify that the program is executable at each step

## Configuration File Structure
The `config.py` file will contain:
1. A list of configuration profiles
2. Each profile will contain all parameters needed for the evaluation
3. Parameters will include data paths, experiment settings, and execution options

## Migration Strategy
The migration will be done in a step-by-step manner:
1. First, create the new configuration system alongside the existing CLI
2. Ensure the new system works with a minimal configuration
3. Gradually migrate all functionality from the CLI to the configuration system
4. Once all functionality is migrated, remove the old CLI code
5. Update all documentation to reflect the new approach

## Testing Approach
1. Create a minimal test configuration that can run the entire program from start to finish
2. Test each step of the migration to ensure functionality is preserved
3. Validate that the program is executable at each step
4. Ensure all existing functionality is maintained

## Expected Outcomes
1. A configuration-file-based system that is easier to use and maintain
2. Better support for multiple configuration profiles
3. Improved reproducibility of experiments
4. Simplified execution process
