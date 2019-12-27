from os import path
import tvm
from ctypes import CDLL, RTLD_GLOBAL


def load_datatype(dtype,
                  code,
                  library_path,
                  casts_from_this_type_map,
                  casts_to_this_type_map,
                  op_map,
                  intrinsic_map,
                  minimum_func=None):
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
    minimum_func: minimum function needed by
      tvm.datatype.register_min_func
    """

    tvm.datatype.register(dtype, code)
    CDLL(library_path, RTLD_GLOBAL)

    for src_dtype, lowering_func in casts_from_this_type_map.items():
        tvm.datatype.register_op(lowering_func, "Cast", "llvm", dtype,
                                 src_dtype)

    for dest_dtype, lowering_func in casts_to_this_type_map.items():
        tvm.datatype.register_op(lowering_func, "Cast", "llvm", dest_dtype,
                                 dtype)

    for op_name, lowering_func in op_map.items():
        tvm.datatype.register_op(lowering_func, op_name, "llvm", dtype)

    for intrinsic_name, lowering_func in intrinsic_map.items():
        tvm.datatype.register_op(lowering_func,
                                 "Call",
                                 "llvm",
                                 dtype,
                                 intrinsic_name=intrinsic_name)

    if minimum_func:
        tvm.datatype.register_min_func(minimum_func, dtype)


def load_bfloat():
    library_path = path.join(path.abspath(path.dirname(__file__)),
                             '../../datatypes/bfloat16/bfloat16.so')
    casts_from_this_type_map = {
        'float': tvm.datatype.create_lower_func("BFloat16ToFloat_wrapper"),
    }
    casts_to_this_type_map = {
        'float': tvm.datatype.create_lower_func("FloatToBFloat16_wrapper"),
        'int': tvm.datatype.create_lower_func("IntToBFloat16_wrapper"),
    }
    op_map = {
        'Add': tvm.datatype.create_lower_func("BFloat16Add_wrapper"),
        'Sub': tvm.datatype.create_lower_func("BFloat16Sub_wrapper"),
        'FloatImm': tvm.datatype.create_lower_func("FloatToBFloat16_wrapper"),
        'Mul': tvm.datatype.create_lower_func("BFloat16Mul_wrapper"),
        'Div': tvm.datatype.create_lower_func("BFloat16Div_wrapper"),
        'Max': tvm.datatype.create_lower_func("BFloat16Max_wrapper"),
    }
    intrinsic_map = {
        'sqrt': tvm.datatype.create_lower_func("BFloat16Sqrt_wrapper"),
        'tvm_if_then_else': tvm.datatype.lower_ite,
        'exp': tvm.datatype.create_lower_func("BFloat16Exp_wrapper"),
    }
    # TODO(gus) these aren't actually right. these are double min(actually
    # lowest)/max.
    minimum_func = lambda _: -1.79769e+308

    load_datatype('bfloat',
                  129,
                  library_path,
                  casts_from_this_type_map,
                  casts_to_this_type_map,
                  op_map,
                  intrinsic_map,
                  minimum_func=minimum_func)


def load_posit8():
    library_path = path.join(path.abspath(path.dirname(__file__)),
                             '../../datatypes/universal-wrapper/universal-wrapper.so')
    casts_from_this_type_map = {
        'float': tvm.datatype.create_lower_func("Posit8es0ToFloat"),
    }
    casts_to_this_type_map = {
        'float': tvm.datatype.create_lower_func("FloatToPosit8es0"),
        'int': tvm.datatype.create_lower_func("IntToPosit8es0"),
    }
    op_map = {
        'Add': tvm.datatype.create_lower_func("Posit8es0Add"),
        'Sub': tvm.datatype.create_lower_func("Posit8es0Sub"),
        'FloatImm': tvm.datatype.create_lower_func("FloatToPosit8es0"),
        'Mul': tvm.datatype.create_lower_func("Posit8es0Mul"),
        'Div': tvm.datatype.create_lower_func("Posit8es0Div"),
        'Max': tvm.datatype.create_lower_func("Posit8es0Max"),
    }
    intrinsic_map = {
        'sqrt': tvm.datatype.create_lower_func("Posit8es0Sqrt"),
        'tvm_if_then_else': tvm.datatype.lower_ite,
        'exp': tvm.datatype.create_lower_func("Posit8es0Exp"),
    }
    # TODO(gus) these aren't actually right. these are double min(actually
    # lowest)/max.
    minimum_func = lambda _: -1.79769e+308

    load_datatype('posit8',
                  130,
                  library_path,
                  casts_from_this_type_map,
                  casts_to_this_type_map,
                  op_map,
                  intrinsic_map,
                  minimum_func=minimum_func)


def load_posit16():
    library_path = path.join(path.abspath(path.dirname(__file__)),
                             '../../datatypes/universal-wrapper/universal-wrapper.so')
    casts_from_this_type_map = {
        'float': tvm.datatype.create_lower_func("Posit16es1ToFloat"),
    }
    casts_to_this_type_map = {
        'float': tvm.datatype.create_lower_func("FloatToPosit16es1"),
        'int': tvm.datatype.create_lower_func("IntToPosit16es1"),
    }
    op_map = {
        'Add': tvm.datatype.create_lower_func("Posit16es1Add"),
        'Sub': tvm.datatype.create_lower_func("Posit16es1Sub"),
        'FloatImm': tvm.datatype.create_lower_func("FloatToPosit16es1"),
        'Mul': tvm.datatype.create_lower_func("Posit16es1Mul"),
        'Div': tvm.datatype.create_lower_func("Posit16es1Div"),
        'Max': tvm.datatype.create_lower_func("Posit16es1Max"),
    }
    intrinsic_map = {
        'sqrt': tvm.datatype.create_lower_func("Posit16es1Sqrt"),
        'tvm_if_then_else': tvm.datatype.lower_ite,
        'exp': tvm.datatype.create_lower_func("Posit16es1Exp"),
    }
    # TODO(gus) these aren't actually right. these are double min(actually
    # lowest)/max.
    minimum_func = lambda _: -1.79769e+308

    load_datatype('posit16',
                  131,
                  library_path,
                  casts_from_this_type_map,
                  casts_to_this_type_map,
                  op_map,
                  intrinsic_map,
                  minimum_func=minimum_func)


def load_posit32():
    library_path = path.join(path.abspath(path.dirname(__file__)),
                             '../../datatypes/universal-wrapper/universal-wrapper.so')
    casts_from_this_type_map = {
        'float': tvm.datatype.create_lower_func("Posit32es2ToFloat"),
    }
    casts_to_this_type_map = {
        'float': tvm.datatype.create_lower_func("FloatToPosit32es2"),
        'int': tvm.datatype.create_lower_func("IntToPosit32es2"),
    }
    op_map = {
        'Add': tvm.datatype.create_lower_func("Posit32es2Add"),
        'Sub': tvm.datatype.create_lower_func("Posit32es2Sub"),
        'FloatImm': tvm.datatype.create_lower_func("FloatToPosit32es2"),
        'Mul': tvm.datatype.create_lower_func("Posit32es2Mul"),
        'Div': tvm.datatype.create_lower_func("Posit32es2Div"),
        'Max': tvm.datatype.create_lower_func("Posit32es2Max"),
    }
    intrinsic_map = {
        'sqrt': tvm.datatype.create_lower_func("Posit32es2Sqrt"),
        'tvm_if_then_else': tvm.datatype.lower_ite,
        'exp': tvm.datatype.create_lower_func("Posit32es2Exp"),
    }
    # TODO(gus) these aren't actually right. these are double min(actually
    # lowest)/max.
    minimum_func = lambda _: -1.79769e+308

    load_datatype('posit32',
                  131,
                  library_path,
                  casts_from_this_type_map,
                  casts_to_this_type_map,
                  op_map,
                  intrinsic_map,
                  minimum_func=minimum_func)
