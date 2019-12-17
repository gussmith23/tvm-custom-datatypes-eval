import tvm
from tvm import relay
from ctypes import CDLL, RTLD_GLOBAL


def convert_ndarray(dst_dtype, array, executor):
    """Converts an NDArray into the specified datatype"""
    x = relay.var('x', shape=array.shape, dtype=str(array.dtype))
    cast = relay.Function([x], x.astype(dst_dtype))
    return executor.evaluate(cast)(array)


def change_dtype(src, dst, expr, params, executor):
    """Change the datatype of a relay expression"""
    # TODO(gus) There's probably a better way to do this---update cdtype to work
    # over modules
    expr = transform.InferType()(relay.Module.from_expr(expr))
    cdtype = relay.frontend.ChangeDatatype(src, dst)
    expr = cdtype.visit(expr['main'])
    expr = transform.InferType()(relay.Module.from_expr(expr))
    expr = expr['main']
    params = dict(
        (p, convert_ndarray(dst, params[p], executor)) for p in params)
    return expr, params


def setup_datatype(dtype, code, library_path, casts_from_this_type_map,
                   casts_to_this_type_map, op_map, intrinsic_map):
    """Utility function for registering a datatype.

    dtype: the type name to register
    code: the code to register the datatype with (>=kCustomBegin, which
    is 129 at the moment)
    library_path: path the library implementing this datatype
    casts_from_this_type_map: dictionary mapping dest type names to
      lowering functions which implement the cast from this type to the
      destination type
    casts_to_this_type_map: dictionary mapping source type names to
      lowering functions which implement the cast from the source type to
      this type
    op_map: dictionary mapping op names ("Add", "Sub") to lowering funcs
      similar to above
    intrinsic_map: dictionary mapping intrinsic names ("sqrt") to
      lowering funcs similar to above
    """

    tvm.datatype.register(dtype, code)
    CDLL(library_path, RTLD_GLOBAL)

    for src_dtype, lowering_func in casts_from_this_type_map:
        tvm.datatype.register_op(lowering_func, "Cast", "llvm", dtype,
                                 src_dtype)

    for dest_dtype, lowering_func in casts_to_this_type_map:
        tvm.datatype.register_op(lowering_func, "Cast", "llvm", dest_dtype,
                                 dtype)

    for op_name, lowering_func in op_map:
        tvm.datatype.register_op(lowering_func, op_name, "llvm", dtype)

    for intrinsic_name, lowering_func in intrinsic_map:
        tvm.datatype.register_op(lowering_func,
                                 "Call",
                                 "llvm",
                                 dtype,
                                 intrinsic_name=intrinsic_name)
