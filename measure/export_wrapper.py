import os
import sys
import runpy


MODEL_TYPE = os.environ.get("MODEL_TYPE", "vision")
if MODEL_TYPE != "radioml" and MODEL_TYPE != "language" and MODEL_TYPE != "vision":
    MODEL_TYPE = "vision"
    print("Defaulting Model Type to vision model.")

MODULE_MAP = {
    "radioml": "measure.export_radioml",
    "vision": "measure.export_vision",
    "language": "measure.export_language",
}

if MODEL_TYPE not in MODULE_MAP:
    print(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    print(f"Available: {list(MODULE_MAP.keys())}")
    sys.exit(1)

module_name = MODULE_MAP[MODEL_TYPE]
print(f"Starting export for: {MODEL_TYPE}")
if MODEL_TYPE == "vision":
    runpy.run_module("measure.data_to_numpy", run_name="__main__")
runpy.run_module(module_name, run_name="__main__")
