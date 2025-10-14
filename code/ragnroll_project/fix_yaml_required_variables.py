#!/usr/bin/env python3
"""
Script to automatically add required_variables to YAML configuration files.
This script scans all YAML files in the configs directory and adds required_variables
to PromptBuilder components that don't have them set.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
import re


def find_yaml_files(config_dir: Path) -> List[Path]:
    """Find all YAML files in the config directory recursively."""
    yaml_files = []
    for root, dirs, files in os.walk(config_dir):
        for file in files:
            if file.endswith(('.yaml', '.yml')):
                yaml_files.append(Path(root) / file)
    return yaml_files


def has_required_variables(prompt_builder_config: Dict[str, Any]) -> bool:
    """Check if a PromptBuilder component already has required_variables set."""
    init_params = prompt_builder_config.get('init_parameters', {})
    return 'required_variables' in init_params


def extract_variables_from_template(template: str) -> List[str]:
    """Extract variable names from a Jinja2 template string."""
    # Find all variables in {{variable}} format
    variables = re.findall(r'\{\{\s*(\w+)\s*\}\}', template)

    # Remove duplicates while preserving order
    seen = set()
    unique_variables = []
    for var in variables:
        if var not in seen:
            seen.add(var)
            unique_variables.append(var)

    return unique_variables


def add_required_variables_to_prompt_builder(prompt_builder_config: Dict[str, Any]) -> bool:
    """
    Add required_variables to a PromptBuilder component if missing.
    Returns True if changes were made, False otherwise.
    """
    if has_required_variables(prompt_builder_config):
        return False

    init_params = prompt_builder_config.get('init_parameters', {})
    template = init_params.get('template', '')

    if not template:
        return False

    # Extract variables from template
    variables = extract_variables_from_template(template)

    if variables:
        # Add required_variables to init_parameters
        init_params['required_variables'] = variables
        return True

    return False


def process_yaml_file(file_path: Path) -> bool:
    """
    Process a single YAML file and add required_variables where needed.
    Returns True if the file was modified, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    if not config or not isinstance(config, dict) or 'components' not in config:
        return False

    modified = False
    components = config['components']

    if not isinstance(components, dict):
        print(f"Skipping {file_path}: components is not a dict")
        return False

    # Process each component
    for component_name, component_config in components.items():
        if isinstance(component_config, dict) and component_config.get('type') == 'haystack.components.builders.prompt_builder.PromptBuilder':
            if add_required_variables_to_prompt_builder(component_config):
                print(f"Added required_variables to {component_name} in {file_path}")
                modified = True

    if modified:
        try:
            # Write back the modified config
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
            print(f"Updated {file_path}")
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False

    return modified


def main():
    """Main function to process all YAML files."""
    script_dir = Path(__file__).parent
    config_dir = script_dir / "configs"

    if not config_dir.exists():
        print(f"Config directory not found: {config_dir}")
        return

    print(f"Scanning for YAML files in {config_dir}")
    yaml_files = find_yaml_files(config_dir)

    if not yaml_files:
        print("No YAML files found.")
        return

    print(f"Found {len(yaml_files)} YAML files")

    modified_count = 0
    for yaml_file in yaml_files:
        if process_yaml_file(yaml_file):
            modified_count += 1

    print(f"\nProcessing complete. Modified {modified_count} files.")


if __name__ == "__main__":
    main()
