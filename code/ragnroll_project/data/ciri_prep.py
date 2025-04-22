## Shot Selection
# args.system = "alluxio" | "docker" ...
# args.shot_selection = "random"
# args.validconfig_shot_num = 1
# args.misconfig_shot_num = 3
# args.input_file_content = <file_content>

prompt = """Question: Are there any mistakes in the above configuration file for {system} version {version}? Respond in a json format similar to the following:
{{
    "hasError": boolean, // true if there are errors, false if there are none.
    "errParameter": [], // List containing properties with errors. If there are no errors, leave this as an empty array.
    "reason": [] // List containing explanations for each error. If there are no errors, leave this as an empty array.
}}

Answer:
```json"""
versions = {
    "alluxio": "2.5.0",
    "django": "4.0.0",
    "etcd": "3.5.0",
    "hbase": "2.2.7",
    "hcommon": "3.3.0",
    "hdfs": "3.3.0",
    "postgresql": "13.0",
    "redis": "7.0.0",
    "yarn": "3.3.0",
    "zookeeper": "3.7.0",
}

import json

import sys
from pathlib import Path
print(Path(__file__).parent.parent.parent.parent / "ciri")

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ciri"))

from ciri.pre_processing.shot.shot_selection import ShotSelection

data_paths = [
    "data/processed/config_val_ciri/evaluation_data.json",
    "data/processed/config_val_ciri/val/evaluation_data.json",
    "data/processed/config_val_ciri/test/evaluation_data.json"
]

for data_path in data_paths:
    with open(data_path, "r") as f:
        eval_data = json.load(f)

    new_eval_data = {"test_cases": []}
    for test_case in eval_data["test_cases"]:

        class Args:
            def __init__(self):
                self.system = test_case["meta"]["category"]
                self.shot_system = None
                self.shot_selection = "random"
                self.validconfig_shot_num = 1
                self.misconfig_shot_num = 3

        args = Args()
        args.shot_system = args.system
        shot_selection = ShotSelection(
            args=args,
            input_file_content=test_case["input"]
        )
            
        shot_selection = shot_selection.select()
        test_case["input"] = shot_selection + "\n" + test_case["input"] + "\n" + prompt.format(system=args.system, version=versions[args.system])

        new_eval_data["test_cases"].append(test_case)


    with open(data_path, "w") as f:
        json.dump(new_eval_data, f, indent=4)
