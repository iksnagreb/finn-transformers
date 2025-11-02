# Also make QONNX related passes available when importing this file, these are
# non-default passes
import onnx_passes.passes.imports.qonnx  # noqa: Passes used via registry
import onnx_passes.passes.inline.qonnx  # noqa: Passes used via registry

# TODO: Implement custom and ad hoc passes here...
