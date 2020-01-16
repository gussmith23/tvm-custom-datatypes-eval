import tvm
from tvm import relay
import torchvision
from models.mobilenet_pytorch.load import load_mobilenet
import numpy as np
from sys import stderr
from util.change_dtype import change_dtype, convert_ndarray
from ctypes import CDLL, RTLD_GLOBAL
from os import path

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
tvm.datatype.register("posit16", 131)
tvm.datatype.register_op(tvm.datatype.create_lower_func("_FloatToPosit16es1"),
                         "Cast", "llvm", "posit16", "float")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit16es1ToFloat"),
                         "Cast", "llvm", "float", "posit16")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_IntToPosit16es1"),
                         "Cast", "llvm", "posit16", "int")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit16es1Add"),
                         "Add", "llvm", "posit16")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_Posit16es1Sub"),
                         "Sub", "llvm", "posit16")
tvm.datatype.register_op(tvm.datatype.create_lower_func("_FloatToPosit16es1"),
                         "FloatImm", "llvm", "posit16")
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

# Change the datatype
conversion_executor = relay.create_executor()
expr, params = change_dtype('float32', 'custom[posit16]16', module['main'],
                            params, conversion_executor)
module = relay.module.Module.from_expr(expr)

ex = relay.create_executor(mod=module)
mobilenet = ex.evaluate()

tested = 0
correct = 0
for image, target_class in dataset:
    # Add batch dimension
    image_tvm = np.expand_dims(image.numpy().astype('float32'), axis=0)

    # Change datatype of input
    image_tvm = convert_ndarray('custom[posit16]16', image_tvm,
                                conversion_executor)

    with tvm.build_config(disable_vectorize=True):
        output = mobilenet(image_tvm, **params)
        output = output.data

    # convert output
    output = convert_ndarray('float32', output, conversion_executor)

    argmax_tvm = np.argmax(output.asnumpy())

    tested += 1
    if (argmax_tvm == target_class): correct += 1
    print('{} of {} correct ({})'.format(correct, tested, correct / tested),
          file=stderr)

print('Model accuracy:')
print(correct / tested)
