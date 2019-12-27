"""Utilities for changing datatypes."""
import tvm
from tvm import relay
from tvm.relay import transform


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
