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

# Load the datatype manually
CDLL(
    path.join(path.abspath(path.dirname(__file__)),
              '../datatypes/nop-type/nop-type.so'), RTLD_GLOBAL)
tvm.datatype.register("nop32", 129)
tvm.datatype.register_op(
    tvm.datatype.create_lower_func("FloatToNop32"), "Cast", "llvm",
    "nop32", "float")
tvm.datatype.register_op(
    tvm.datatype.create_lower_func("Nop32ToFloat"), "Cast", "llvm",
    "float", "nop32")
tvm.datatype.register_op(
    tvm.datatype.create_lower_func("IntToNop32"), "Cast", "llvm",
    "nop32", "int")
tvm.datatype.register_op(
    tvm.datatype.create_lower_func("Nop32Add"), "Add", "llvm",
    "nop32")
tvm.datatype.register_op(
    tvm.datatype.create_lower_func("Nop32Sub"), "Sub", "llvm",
    "nop32")
tvm.datatype.register_op(
    tvm.datatype.create_lower_func("FloatToNop32"), "FloatImm",
    "llvm", "nop32")
tvm.datatype.register_op(
    tvm.datatype.create_lower_func("Nop32Mul"), "Mul", "llvm",
    "nop32")
tvm.datatype.register_op(
    tvm.datatype.create_lower_func("Nop32Div"), "Div", "llvm",
    "nop32")
tvm.datatype.register_op(
    tvm.datatype.create_lower_func("Nop32Max"), "Max", "llvm",
    "nop32")
tvm.datatype.register_op(
    tvm.datatype.create_lower_func("Nop32Sqrt"),
    "Call",
    "llvm",
    "nop32",
    intrinsic_name="sqrt")
tvm.datatype.register_op(tvm.datatype.lower_ite,
                         "Call",
                         "llvm",
                         "nop32",
                         intrinsic_name="tvm_if_then_else")
tvm.datatype.register_op(
    tvm.datatype.create_lower_func("Nop32Exp"),
    "Call",
    "llvm",
    "nop32",
    intrinsic_name="exp")
tvm.datatype.register_min_func(lambda num_bits: -3.38953139e38, "nop32")

# Change the datatype
conversion_executor = relay.create_executor()
expr, params = change_dtype('float32', 'custom[nop32]32', module['main'],
                            params, conversion_executor)
module = relay.module.Module.from_expr(expr)

run_pretrained_model_test(module, params, dataset, 'custom[nop32]32')
