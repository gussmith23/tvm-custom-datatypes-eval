import tvm
from tvm import relay
import torchvision
from models.mobilenet_pytorch.load import load_mobilenet
import numpy as np
from util.load_datatypes import load_posit32
from util.change_dtype import change_dtype, convert_ndarray
from ctypes import CDLL, RTLD_GLOBAL
from os import path
from util.run_pretrained_model_test import run_pretrained_model_test

# Copied from https://github.com/kuangliu/pytorch-cifar/blob/ab908327d44bf9b1d22cd333a4466e85083d3f21/main.py#L36
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
])
# The repo I used to generate the trained params also uses torchvision's data
# loader, but with train=True.
dataset = torchvision.datasets.CIFAR10('.',
                                       train=False,
                                       download=True,
                                       transform=transform)

module, params, image_shape = load_mobilenet()

# Register datatype manually
CDLL(
    path.join(path.abspath(path.dirname(__file__)),
              '../datatypes/universal-wrapper/universal-wrapper.so'),
    RTLD_GLOBAL)
tvm.datatype.register("posit32", 131)
tvm.datatype.register_op(tvm.datatype.create_lower_func("_FloatToPosit32es2"),
                         "Cast", "llvm", "posit32", "float")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit32es2ToFloat"),
                         "Cast", "llvm", "float", "posit32")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_IntToPosit32es2"),
                         "Cast", "llvm", "posit32", "int")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit32es2Add"),
                         "Add", "llvm", "posit32")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit32es2Sub"),
                         "Sub", "llvm", "posit32")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_FloatToPosit32es2"),
                         "FloatImm", "llvm", "posit32")
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

# Change the datatype
conversion_executor = relay.create_executor()
expr, params = change_dtype('float32', 'custom[posit32]32', module['main'],
                            params, conversion_executor)
module = relay.module.Module.from_expr(expr)

run_pretrained_model_test(module, params, dataset, 'custom[posit32]32')
