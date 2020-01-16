import tvm
from tvm import relay
import torchvision
from models.mobilenet_pytorch.load import load_mobilenet
import numpy as np
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
tvm.datatype.register("posit8", 131)
tvm.datatype.register_op(tvm.datatype.create_lower_func("_FloatToPosit8es0"),
                         "Cast", "llvm", "posit8", "float")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit8es0ToFloat"),
                         "Cast", "llvm", "float", "posit8")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_IntToPosit8es0"),
                         "Cast", "llvm", "posit8", "int")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit8es0Add"),
                         "Add", "llvm", "posit8")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit8es0Sub"),
                         "Sub", "llvm", "posit8")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_FloatToPosit8es0"),
                         "FloatImm", "llvm", "posit8")
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

# Change the datatype
conversion_executor = relay.create_executor()
expr, params = change_dtype('float32', 'custom[posit8]8', module['main'],
                            params, conversion_executor)
module = relay.module.Module.from_expr(expr)

run_pretrained_model_test(module, params, dataset, 'custom[posit8]8')
