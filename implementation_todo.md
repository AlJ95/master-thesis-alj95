# RAGnRoll Implementation Todo List

## Phase 1: Create the New Configuration System
- [x] Create a `main.py` file that will be the new entry point
- [x] Create a `config.py` file that will contain all configuration parameters
- [x] Implement a configuration parser that can handle multiple profiles
- [x] Create a mapping between CLI parameters and configuration parameters

## Phase 2: Migrate CLI Functionality
- [ ] Migrate the `run_evaluations` command functionality to use the configuration file
- [ ] Migrate the `split_data` command functionality
- [ ] Migrate the `test_generalization_error` command functionality
- [ ] Migrate the `draw_pipeline` command functionality

## Phase 3: Testing and Validation
- [ ] Create a minimal test configuration file
- [ ] Ensure the program is executable at each step
- [ ] Validate that all functionality is preserved
- [ ] Create documentation for the new configuration approach

## Phase 4: Documentation and Cleanup
- [ ] Remove the old CLI code
- [ ] Update documentation to reflect the new approach
- [ ] Create examples of configuration files

## Detailed Implementation Steps

### Step 1: Enhance the main.py entry point
- [ ] Add proper error handling
- [ ] Add logging support
- [ ] Add support for command-line arguments (optional, for backward compatibility)

### Step 2: Design the config.py structure
- [ ] Define a structure that can contain all the parameters currently passed via CLI
- [ ] Support multiple configuration profiles in a list format
- [ ] Include all necessary parameters for data paths, experiment settings, etc.

### Step 3: Implement configuration loading
- [ ] Create a function to load and validate the configuration file
- [ ] Implement profile selection and iteration logic
- [ ] Add validation for required parameters

### Step 4: Migrate run_evaluations functionality
- [ ] Adapt the existing `run_evaluations` function to use configuration parameters instead of CLI arguments
- [ ] Ensure all current functionality is preserved
- [ ] Add support for all existing CLI options

### Step 5: Migrate other CLI commands
- [ ] Adapt `split_data`, `test_generalization_error`, and `draw_pipeline` functions to use configuration parameters
- [ ] Ensure backward compatibility where needed

### Step 6: Create test configuration
- [ ] Create a minimal `config.py` file that can be used for testing
- [ ] Ensure it works with a small dataset from start to finish
- [ ] Add example configurations for different use cases

### Step 7: Validate and test
- [ ] Test the new system with the minimal configuration
- [ ] Ensure all functionality works as expected
- [ ] Verify that the program is executable at each step

## Configuration File Structure
- [ ] Define the complete structure for the configuration file
- [ ] Document all available parameters
- [ ] Add support for environment variables
- [ ] Add support for default values

## Migration Strategy
- [ ] First, create the new configuration system alongside the existing CLI
- [ ] Ensure the new system works with a minimal configuration
- [ ] Gradually migrate all functionality from the CLI to the configuration system
- [ ] Once all functionality is migrated, remove the old CLI code
- [ ] Update all documentation to reflect the new approach

## Testing Approach
- [ ] Create a minimal test configuration that can run the entire program from start to finish
- [ ] Test each step of the migration to ensure functionality is preserved
- [ ] Validate that the program is executable at each step
- [ ] Ensure all existing functionality is maintained
