# Also make QONNX related passes available when importing this file, these are
# non-default passes
import onnx_passes.passes.imports.qonnx  # noqa: Passes used via registry
import onnx_passes.passes.inline.qonnx  # noqa: Passes used via registry

# IThe passes module sets up the registry and makes the @passes.register
# decorator work
import onnx_passes.passes as passes


# Custom cleanup pass to wrap a sequence of cleanup and annotation passes in an
# exhaustive manner
@passes.verify.tolerance
@passes.register("tidyup")
class Tidyup(passes.base.Transformation, passes.compose.ComposePass):
    __passes__ = [
        "shape-inference",
        "fold-constants",
        "eliminate",
        "cleanup",
    ]

    __exhaustive__ = True
