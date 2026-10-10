"""Frontend: input validation and the kernel ABI.

SAVA kernels receive every input *and* output as a DRAM buffer (registers
a0..a7). Frontends (torch/onnx/stablehlo importers, hand-written MLIR) instead
produce functions that *return* tensors, so the first step rewrites

    func.func @f(%a: tensor<..>) -> tensor<..> { ...; return %r }

into destination-passing style (DPS):

    func.func @f(%a: tensor<..>, %out: memref<..>) {
      ...
      bufferization.materialize_in_destination %r in restrict writable %out
      return
    }

Together with `--eliminate-empty-tensors` this lets bufferization compute
results directly into the output buffer, without a temporary and a copy. This
is the same idea as IREE's dispatch-region ABI (flow.dispatch.tensor.store
into a binding).
"""

from __future__ import annotations

from dataclasses import dataclass

from iree.compiler import ir

SUPPORTED_ELEMENT_TYPES = ("f32", "bf16", "i32", "i8")


class FrontendError(Exception):
    pass


@dataclass
class KernelSignature:
    name: str
    num_inputs: int
    num_outputs: int


def find_kernel(module: ir.Module) -> ir.Operation:
    funcs = [op for op in module.body.operations
             if op.operation.name == "func.func" and len(op.regions[0].blocks) > 0]
    public = [f for f in funcs
              if "sym_visibility" not in f.attributes
              or ir.StringAttr(f.attributes["sym_visibility"]).value == "public"]
    if len(public) != 1:
        raise FrontendError(f"expected exactly one public function with a body, found {len(public)}")
    return public[0]


def _check_tensor(t: ir.Type, what: str) -> ir.RankedTensorType:
    if not isinstance(t, ir.RankedTensorType):
        raise FrontendError(f"{what}: only ranked tensors are supported, got {t}")
    rt = ir.RankedTensorType(t)
    if not rt.has_static_shape:
        raise FrontendError(f"{what}: dynamic shapes are not supported yet, got {t}")
    if str(rt.element_type) not in SUPPORTED_ELEMENT_TYPES:
        raise FrontendError(f"{what}: element type {rt.element_type} is not supported")
    return rt


def to_destination_passing_style(module: ir.Module) -> KernelSignature:
    """Rewrites the (single) public function in place. See module docstring."""
    fn = find_kernel(module)
    name = ir.StringAttr(fn.attributes["sym_name"]).value
    ftype = ir.FunctionType(ir.TypeAttr(fn.attributes["function_type"]).value)
    inputs, results = list(ftype.inputs), list(ftype.results)
    for i, t in enumerate(inputs):
        _check_tensor(t, f"argument {i}")
    if not results:
        raise FrontendError("the kernel function has no results")

    block = fn.regions[0].blocks[0]
    ret = block.operations[len(block.operations) - 1]
    if ret.operation.name != "func.return":
        raise FrontendError("expected func.return terminator")

    with ir.InsertionPoint(ret), ret.location:
        out_types = []
        for i, (t, value) in enumerate(zip(results, ret.operands)):
            rt = _check_tensor(t, f"result {i}")
            mt = ir.MemRefType.get(rt.shape, rt.element_type)
            out_types.append(mt)
            out_arg = block.add_argument(mt, ret.location)
            ir.Operation.create(
                "bufferization.materialize_in_destination",
                results=[],
                operands=[value, out_arg],
                attributes={"restrict": ir.UnitAttr.get(), "writable": ir.UnitAttr.get()},
            )
        ir.Operation.create("func.return", results=[], operands=[])
    ret.erase()

    fn.attributes["function_type"] = ir.TypeAttr.get(ir.FunctionType.get(inputs + out_types, []))
    return KernelSignature(name, len(inputs), len(results))
