from os import path
import tvm
from ctypes import CDLL, RTLD_GLOBAL


def load_nop32():
    # Load the datatype manually
    CDLL(
        path.join(path.abspath(path.dirname(__file__)),
                  '../../datatypes/nop-type/nop-type.so'), RTLD_GLOBAL)
    tvm.datatype.register("nop32", 129)
    tvm.datatype.register_op(tvm.datatype.create_lower_func("FloatToNop32"),
                             "Cast", "llvm", "nop32", "float")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Nop32ToFloat"),
                             "Cast", "llvm", "float", "nop32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("IntToNop32"),
                             "Cast", "llvm", "nop32", "int")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Nop32Add"), "Add",
                             "llvm", "nop32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Nop32Sub"), "Sub",
                             "llvm", "nop32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("FloatToNop32"),
                             "FloatImm", "llvm", "nop32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Nop32Mul"), "Mul",
                             "llvm", "nop32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Nop32Div"), "Div",
                             "llvm", "nop32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Nop32Max"), "Max",
                             "llvm", "nop32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Nop32Sqrt"),
                             "Call",
                             "llvm",
                             "nop32",
                             intrinsic_name="sqrt")
    tvm.datatype.register_op(tvm.datatype.lower_ite,
                             "Call",
                             "llvm",
                             "nop32",
                             intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Nop32Exp"),
                             "Call",
                             "llvm",
                             "nop32",
                             intrinsic_name="exp")
    tvm.datatype.register_min_func(lambda num_bits: -3.38953139e38, "nop32")


def load_bfloat16():
    # Load the datatype manually
    CDLL(
        path.join(path.abspath(path.dirname(__file__)),
                  '../../datatypes/bfloat16/bfloat16.so'), RTLD_GLOBAL)
    tvm.datatype.register("bfloat16", 129)
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToBFloat16_wrapper"), "Cast",
        "llvm", "bfloat16", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_BFloat16ToFloat_wrapper"), "Cast",
        "llvm", "float", "bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_IntToBFloat16_wrapper"), "Cast",
        "llvm", "bfloat16", "int")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_BFloat16Add_wrapper"), "Add", "llvm",
        "bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_BFloat16Sub_wrapper"), "Sub", "llvm",
        "bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToBFloat16_wrapper"), "FloatImm",
        "llvm", "bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_BFloat16Mul_wrapper"), "Mul", "llvm",
        "bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_BFloat16Div_wrapper"), "Div", "llvm",
        "bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_BFloat16Max_wrapper"), "Max", "llvm",
        "bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_BFloat16Sqrt_wrapper"),
        "Call",
        "llvm",
        "bfloat16",
        intrinsic_name="sqrt")
    tvm.datatype.register_op(tvm.datatype.lower_ite,
                             "Call",
                             "llvm",
                             "bfloat16",
                             intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_BFloat16Exp_wrapper"),
        "Call",
        "llvm",
        "bfloat16",
        intrinsic_name="exp")
    tvm.datatype.register_min_func(lambda num_bits: -3.38953139e38, "bfloat16")


def load_posit8():
    # Register datatype manually
    CDLL(
        path.join(path.abspath(path.dirname(__file__)),
                  '../../datatypes/universal-wrapper/universal-wrapper.so'),
        RTLD_GLOBAL)
    tvm.datatype.register("posit8", 131)
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToPosit8es0"), "Cast", "llvm",
        "posit8", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_Posit8es0ToFloat"), "Cast", "llvm",
        "float", "posit8")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_IntToPosit8es0"),
                             "Cast", "llvm", "posit8", "int")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit8es0Add"),
                             "Add", "llvm", "posit8")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit8es0Sub"),
                             "Sub", "llvm", "posit8")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToPosit8es0"), "FloatImm",
        "llvm", "posit8")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit8es0Mul"),
                             "Mul", "llvm", "posit8")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit8es0Div"),
                             "Div", "llvm", "posit8")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit8es0Max"),
                             "Max", "llvm", "posit8")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit8es0Sqrt"),
                             "Call",
                             "llvm",
                             "posit8",
                             intrinsic_name="sqrt")
    tvm.datatype.register_op(tvm.datatype.lower_ite,
                             "Call",
                             "llvm",
                             "posit8",
                             intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit8es0Exp"),
                             "Call",
                             "llvm",
                             "posit8",
                             intrinsic_name="exp")
    # es = 0, useed = 2. first bit is sign bit, then 7 bits of '1' for the regime
    # gives us a k value of 6. Then our number is 2**6.
    tvm.datatype.register_min_func(lambda num_bits: -64, "posit8")


def load_posit16():
    # Register datatype manually
    CDLL(
        path.join(path.abspath(path.dirname(__file__)),
                  '../../datatypes/universal-wrapper/universal-wrapper.so'),
        RTLD_GLOBAL)
    tvm.datatype.register("posit16", 131)
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToPosit16es1"), "Cast", "llvm",
        "posit16", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_Posit16es1ToFloat"), "Cast", "llvm",
        "float", "posit16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_IntToPosit16es1"), "Cast", "llvm",
        "posit16", "int")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit16es1Add"),
                             "Add", "llvm", "posit16")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit16es1Sub"),
                             "Sub", "llvm", "posit16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToPosit16es1"), "FloatImm",
        "llvm", "posit16")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit16es1Mul"),
                             "Mul", "llvm", "posit16")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit16es1Div"),
                             "Div", "llvm", "posit16")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit16es1Max"),
                             "Max", "llvm", "posit16")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit16es1Sqrt"),
                             "Call",
                             "llvm",
                             "posit16",
                             intrinsic_name="sqrt")
    tvm.datatype.register_op(tvm.datatype.lower_ite,
                             "Call",
                             "llvm",
                             "posit16",
                             intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit16es1Exp"),
                             "Call",
                             "llvm",
                             "posit16",
                             intrinsic_name="exp")
    # es = 1, useed = 4. first bit is sign bit, then 15 bits of '1' for the regime
    # gives us a k value of 14. Then our number is 4**14
    tvm.datatype.register_min_func(lambda num_bits: -268435456, "posit16")


def load_posit32():
    # Register datatype manually
    CDLL(
        path.join(path.abspath(path.dirname(__file__)),
                  '../../datatypes/universal-wrapper/universal-wrapper.so'),
        RTLD_GLOBAL)
    tvm.datatype.register("posit32", 131)
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToPosit32es2"), "Cast", "llvm",
        "posit32", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_Posit32es2ToFloat"), "Cast", "llvm",
        "float", "posit32")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_IntToPosit32es2"), "Cast", "llvm",
        "posit32", "int")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit32es2Add"),
                             "Add", "llvm", "posit32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit32es2Sub"),
                             "Sub", "llvm", "posit32")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToPosit32es2"), "FloatImm",
        "llvm", "posit32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit32es2Mul"),
                             "Mul", "llvm", "posit32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit32es2Div"),
                             "Div", "llvm", "posit32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit32es2Max"),
                             "Max", "llvm", "posit32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit32es2Sqrt"),
                             "Call",
                             "llvm",
                             "posit32",
                             intrinsic_name="sqrt")
    tvm.datatype.register_op(tvm.datatype.lower_ite,
                             "Call",
                             "llvm",
                             "posit32",
                             intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit32es2Exp"),
                             "Call",
                             "llvm",
                             "posit32",
                             intrinsic_name="exp")
    # es = 2, useed = 16. first bit is sign bit, then 31 bits of '1' for the regime
    # gives us a k value of 30. Then our number is 16**30.
    tvm.datatype.register_min_func(lambda num_bits: -1.329228e+36, "posit32")


def load_float32():
    CDLL(
        path.join(path.abspath(path.dirname(__file__)),
                  '../../datatypes/float32/float32.so'), RTLD_GLOBAL)
    tvm.datatype.register("ourfloat32", 131)
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToOurFloat32"), "Cast", "llvm",
        "ourfloat32", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("OurFloat32ToFloat"), "Cast", "llvm",
        "float", "ourfloat32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("IntToOurFloat32"),
                             "Cast", "llvm", "ourfloat32", "int")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("OurFloat32Add"),
                             "Add", "llvm", "ourfloat32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("OurFloat32Sub"),
                             "Sub", "llvm", "ourfloat32")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToOurFloat32"), "FloatImm",
        "llvm", "ourfloat32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("OurFloat32Mul"),
                             "Mul", "llvm", "ourfloat32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("OurFloat32Div"),
                             "Div", "llvm", "ourfloat32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("OurFloat32Max"),
                             "Max", "llvm", "ourfloat32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("OurFloat32Sqrt"),
                             "Call",
                             "llvm",
                             "ourfloat32",
                             intrinsic_name="sqrt")
    tvm.datatype.register_op(tvm.datatype.lower_ite,
                             "Call",
                             "llvm",
                             "ourfloat32",
                             intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("OurFloat32Exp"),
                             "Call",
                             "llvm",
                             "ourfloat32",
                             intrinsic_name="exp")
    # es = 0, useed = 2. first bit is sign bit, then 7 bits of '1' for the regime
    # gives us a k value of 6. Then our number is 2**6.
    tvm.datatype.register_min_func(lambda num_bits: -3.4028235e+38,
                                   "ourfloat32")


def load_counted_float32():
    dll = CDLL(
        path.join(path.abspath(path.dirname(__file__)),
                  '../../datatypes/counted-float32/counted-float32.so'),
        RTLD_GLOBAL)
    tvm.datatype.register("countedfloat32", 132)
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToCountedFloat32"), "Cast",
        "llvm", "countedfloat32", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("CountedFloat32ToFloat"), "Cast",
        "llvm", "float", "countedfloat32")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("IntToCountedFloat32"), "Cast", "llvm",
        "countedfloat32", "int")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("CountedFloat32Add"), "Add", "llvm",
        "countedfloat32")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("CountedFloat32Sub"), "Sub", "llvm",
        "countedfloat32")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToCountedFloat32"), "FloatImm",
        "llvm", "countedfloat32")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("CountedFloat32Mul"), "Mul", "llvm",
        "countedfloat32")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("CountedFloat32Div"), "Div", "llvm",
        "countedfloat32")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("CountedFloat32Max"), "Max", "llvm",
        "countedfloat32")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("CountedFloat32Sqrt"),
        "Call",
        "llvm",
        "countedfloat32",
        intrinsic_name="sqrt")
    tvm.datatype.register_op(tvm.datatype.lower_ite,
                             "Call",
                             "llvm",
                             "countedfloat32",
                             intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("CountedFloat32Exp"),
        "Call",
        "llvm",
        "countedfloat32",
        intrinsic_name="exp")
    # es = 0, useed = 2. first bit is sign bit, then 7 bits of '1' for the regime
    # gives us a k value of 6. Then our number is 2**6.
    tvm.datatype.register_min_func(lambda num_bits: -3.4028235e+38,
                                   "countedfloat32")

    return dll

def load_libposit_posit8():
    # Register datatype manually
    CDLL(
        path.join(path.abspath(path.dirname(__file__)),
                  '../../datatypes/libposit/libposit-wrapper.so'),
        RTLD_GLOBAL)
    tvm.datatype.register("libposit-posit8", 131)
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToLibPosit_Posit8es0"), "Cast", "llvm",
        "libposit-posit8", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_LibPosit_Posit8es0_ToFloat"), "Cast", "llvm",
        "float", "libposit-posit8")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_IntToLibPosit_Posit8es0"),
                             "Cast", "llvm", "libposit-posit8", "int")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit8es0_Add"),
                             "Add", "llvm", "libposit-posit8")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit8es0_Sub"),
                             "Sub", "llvm", "libposit-posit8")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToLibPosit_Posit8es0"), "FloatImm",
        "llvm", "libposit-posit8")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit8es0_Mul"),
                             "Mul", "llvm", "libposit-posit8")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit8es0_Div"),
                             "Div", "llvm", "libposit-posit8")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit8es0_Max"),
                             "Max", "llvm", "libposit-posit8")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit8es0_Sqrt"),
                             "Call",
                             "llvm",
                             "libposit-posit8",
                             intrinsic_name="sqrt")
    tvm.datatype.register_op(tvm.datatype.lower_ite,
                             "Call",
                             "llvm",
                             "libposit-posit8",
                             intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit8es0_Exp"),
                             "Call",
                             "llvm",
                             "libposit-posit8",
                             intrinsic_name="exp")
    # es = 0, useed = 2. first bit is sign bit, then 7 bits of '1' for the regime
    # gives us a k value of 6. Then our number is 2**6.
    tvm.datatype.register_min_func(lambda num_bits: -64, "libposit-posit8")

def load_libposit_posit16():
    # Register datatype manually
    CDLL(
        path.join(path.abspath(path.dirname(__file__)),
                  '../../datatypes/libposit/libposit-wrapper.so'),
        RTLD_GLOBAL)
    tvm.datatype.register("libposit-posit16", 131)
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToLibPosit_Posit16es1"), "Cast", "llvm",
        "libposit-posit16", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_LibPosit_Posit16es1_ToFloat"), "Cast", "llvm",
        "float", "libposit-posit16")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_IntToLibPosit_Posit16es1"),
                             "Cast", "llvm", "libposit-posit16", "int")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit16es1_Add"),
                             "Add", "llvm", "libposit-posit16")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit16es1_Sub"),
                             "Sub", "llvm", "libposit-posit16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToLibPosit_Posit16es1"), "FloatImm",
        "llvm", "libposit-posit16")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit16es1_Mul"),
                             "Mul", "llvm", "libposit-posit16")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit16es1_Div"),
                             "Div", "llvm", "libposit-posit16")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit16es1_Max"),
                             "Max", "llvm", "libposit-posit16")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit16es1_Sqrt"),
                             "Call",
                             "llvm",
                             "libposit-posit16",
                             intrinsic_name="sqrt")
    tvm.datatype.register_op(tvm.datatype.lower_ite,
                             "Call",
                             "llvm",
                             "libposit-posit16",
                             intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit16es1_Exp"),
                             "Call",
                             "llvm",
                             "libposit-posit16",
                             intrinsic_name="exp")
    # es = 1, useed = 4. first bit is sign bit, then 15 bits of '1' for the regime
    # gives us a k value of 14. Then our number is 4**14
    tvm.datatype.register_min_func(lambda num_bits: -268435456, "libposit-posit16")


def load_libposit_posit32():
    # Register datatype manually
    CDLL(
        path.join(path.abspath(path.dirname(__file__)),
                  '../../datatypes/libposit/libposit-wrapper.so'),
        RTLD_GLOBAL)
    tvm.datatype.register("libposit-posit32", 131)
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("posit32_fromf"), "Cast", "llvm",
        "libposit-posit32", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("posit32_tof"), "Cast", "llvm",
        "float", "libposit-posit32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_IntToLibPosit_Posit32es2"),
                             "Cast", "llvm", "libposit-posit32", "int")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit32es2_Add"),
                             "Add", "llvm", "libposit-posit32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit32es2_Sub"),
                             "Sub", "llvm", "libposit-posit32")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("_FloatToLibPosit_Posit32es2"), "FloatImm",
        "llvm", "libposit-posit32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit32es2_Mul"),
                             "Mul", "llvm", "libposit-posit32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit32es2_Div"),
                             "Div", "llvm", "libposit-posit32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit32es2_Max"),
                             "Max", "llvm", "libposit-posit32")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit32es2_Sqrt"),
                             "Call",
                             "llvm",
                             "libposit-posit32",
                             intrinsic_name="sqrt")
    tvm.datatype.register_op(tvm.datatype.lower_ite,
                             "Call",
                             "llvm",
                             "libposit-posit32",
                             intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("_LibPosit_Posit32es2_Exp"),
                             "Call",
                             "llvm",
                             "libposit-posit32",
                             intrinsic_name="exp")
    # es = 2, useed = 16. first bit is sign bit, then 31 bits of '1' for the regime
    # gives us a k value of 30. Then our number is 16**30.
    tvm.datatype.register_min_func(lambda num_bits: -1.329228e+36, "libposit-posit32")

def load_biovault_bfloat16():
    # Load the datatype manually
    CDLL(
        path.join(path.abspath(path.dirname(__file__)),
                  '../../datatypes/biovault_bfloat16/biovault-bfloat16.so'), RTLD_GLOBAL)
    tvm.datatype.register("biovault-bfloat16", 135)
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToBiovaultBFloat16"), "Cast",
        "llvm", "biovault-bfloat16", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BiovaultBFloat16ToFloat"), "Cast",
        "llvm", "float", "biovault-bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("IntToBiovaultBFloat16"), "Cast",
        "llvm", "biovault-bfloat16", "int")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BiovaultBFloat16Add"), "Add", "llvm",
        "biovault-bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BiovaultBFloat16Sub"), "Sub", "llvm",
        "biovault-bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToBiovaultBFloat16"), "FloatImm",
        "llvm", "biovault-bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BiovaultBFloat16Mul"), "Mul", "llvm",
        "biovault-bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BiovaultBFloat16Div"), "Div", "llvm",
        "biovault-bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BiovaultBFloat16Max"), "Max", "llvm",
        "biovault-bfloat16")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BiovaultBFloat16Sqrt"),
        "Call",
        "llvm",
        "biovault-bfloat16",
        intrinsic_name="sqrt")
    tvm.datatype.register_op(tvm.datatype.lower_ite,
                             "Call",
                             "llvm",
                             "biovault-bfloat16",
                             intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BiovaultBFloat16Exp"),
        "Call",
        "llvm",
        "biovault-bfloat16",
        intrinsic_name="exp")
    tvm.datatype.register_min_func(lambda num_bits: -3.38953139e38, "biovault-bfloat16")
