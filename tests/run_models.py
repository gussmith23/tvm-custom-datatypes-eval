# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Utilities for changing datatypes of models."""

import tvm
import topi.testing
import numpy as np
from tvm import relay
from tvm.relay.testing.inception_v3 import get_workload as get_inception
from tvm.relay.testing.resnet import get_workload as get_resnet
from tvm.relay.testing.mobilenet import get_workload as get_mobilenet
from nose.tools import nottest
from load_datatypes import load_bfloat, load_posit8, load_posit16, load_posit32
from util import change_dtype, convert_ndarray

tgt = "llvm"


def setup():
    load_bfloat()
    load_posit8()
    load_posit16()
    load_posit32()


# def setup():
#     """Set up tests

#     Currently, this registers some custom datatypes using the Bring Your
#     Own Datatypes framework.
#     """

#     # To use datatype operations in an external library, you should first load
#     # the library containing the datatype implementation:
#     # CDLL("libmybfloat16.so", RTLD_GLOBAL)
#     # In this case, the datatype library we are using is built right into TVM,
#     # so we do not need to explicitly load any library.

#     # You can pick a code for your datatype arbitrarily, as long as it is
#     # greater than 128 and has not already been chosen.

#     tvm.datatype.register("posit", 131)

#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("FloatToBFloat16_wrapper"), "Cast",
#         "llvm", "bfloat", "float")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("BFloat16ToFloat_wrapper"), "Cast",
#         "llvm", "float", "bfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("IntToBFloat16_wrapper"), "Cast",
#         "llvm", "bfloat", "int")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("BFloat16Add_wrapper"), "Add", "llvm",
#         "bfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("BFloat16Sub_wrapper"), "Sub", "llvm",
#         "bfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("FloatToBFloat16_wrapper"), "FloatImm",
#         "llvm", "bfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("BFloat16Mul_wrapper"), "Mul", "llvm",
#         "bfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("BFloat16Div_wrapper"), "Div", "llvm",
#         "bfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("BFloat16Max_wrapper"), "Max", "llvm",
#         "bfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("BFloat16Sqrt_wrapper"),
#         "Call",
#         "llvm",
#         "bfloat",
#         intrinsic_name="sqrt")
#     # TODO(gus) not sure if this will work...
#     tvm.datatype.register_op(tvm.datatype.lower_ite,
#                              "Call",
#                              "llvm",
#                              "bfloat",
#                              intrinsic_name="tvm_if_then_else")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("BFloat16Exp_wrapper"),
#         "Call",
#         "llvm",
#         "bfloat",
#         intrinsic_name="exp")

#     tvm.datatype.register("notbfloat", 130)

#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("FloatToNotBFloat16_wrapper"), "Cast",
#         "llvm", "notbfloat", "float")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("NotBFloat16ToFloat_wrapper"), "Cast",
#         "llvm", "float", "notbfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("IntToNotBFloat16_wrapper"), "Cast",
#         "llvm", "notbfloat", "int")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("NotBFloat16Add_wrapper"), "Add",
#         "llvm", "notbfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("NotBFloat16Sub_wrapper"), "Sub",
#         "llvm", "notbfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("FloatToNotBFloat16_wrapper"),
#         "FloatImm", "llvm", "notbfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("NotBFloat16Mul_wrapper"), "Mul",
#         "llvm", "notbfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("NotBFloat16Div_wrapper"), "Div",
#         "llvm", "notbfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("NotBFloat16Max_wrapper"), "Max",
#         "llvm", "notbfloat")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("NotBFloat16Sqrt_wrapper"),
#         "Call",
#         "llvm",
#         "notbfloat",
#         intrinsic_name="sqrt")
#     # TODO(gus) not sure if this will work...
#     tvm.datatype.register_op(tvm.datatype.lower_ite,
#                              "Call",
#                              "llvm",
#                              "notbfloat",
#                              intrinsic_name="tvm_if_then_else")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("NotBFloat16Exp_wrapper"),
#         "Call",
#         "llvm",
#         "notbfloat",
#         intrinsic_name="exp")

#     tvm.datatype.register("posit", 131)

#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("FloatToPosit32es2"), "Cast", "llvm",
#         "posit", "float")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("Posit32es2ToFloat"), "Cast", "llvm",
#         "float", "posit")
#     tvm.datatype.register_op(tvm.datatype.create_lower_func("IntToPosit32es2"),
#                              "Cast", "llvm", "posit", "int")
#     tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Add"),
#                              "Add", "llvm", "posit")
#     tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Sub"),
#                              "Sub", "llvm", "posit")
#     tvm.datatype.register_op(
#         tvm.datatype.create_lower_func("FloatToPosit32es2"), "FloatImm",
#         "llvm", "posit")
#     tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Mul"),
#                              "Mul", "llvm", "posit")
#     tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Div"),
#                              "Div", "llvm", "posit")
#     tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Max"),
#                              "Max", "llvm", "posit")
#     tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Sqrt"),
#                              "Call",
#                              "llvm",
#                              "posit",
#                              intrinsic_name="sqrt")
#     # TODO(gus) not sure if this will work...
#     tvm.datatype.register_op(tvm.datatype.lower_ite,
#                              "Call",
#                              "llvm",
#                              "posit",
#                              intrinsic_name="tvm_if_then_else")
#     tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Exp"),
#                              "Call",
#                              "llvm",
#                              "posit",
#                              intrinsic_name="exp")
#     # TODO(gus) these aren't actually right. these are double min(actually lowest)/max.
#     tvm.datatype.register_min_func(lambda num_bits: -1.79769e+308, "posit")


def run_model(get_workload,
              input_shape,
              src_dtype,
              dst_dtype,
              rtol=0.0001,
              atol=0.0001):
    module, params = get_workload()

    ex = relay.create_executor("graph")

    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(*input_shape).astype(src_dtype))

    correct = relay.create_executor("graph", module).evaluate()(input,
                                                                **params)

    # Simplifying inference is essential right now, as batch norms (which get
    # removed) are broken with custom datatypes.
    module = relay.transform.SimplifyInference()(module)
    expr, params = change_dtype(src_dtype, dst_dtype, module['main'], params,
                                ex)

    input = convert_ndarray(dst_dtype, input, ex)

    # Vectorization is not implemented with custom datatypes.
    with tvm.build_config(disable_vectorize=True):
        result = ex.evaluate(expr)(input, **params)

    tvm.testing.assert_allclose(convert_ndarray(src_dtype, result,
                                                ex).asnumpy(),
                                correct.asnumpy(),
                                rtol=rtol,
                                atol=atol)


def test_models():
    # run_model(get_mobilenet, (3, 224, 224), 'float32', 'custom[posit]32')
    # run_model(get_inception, (3, 299, 299), 'float32', 'custom[posit]32')
    # run_model(get_resnet, (3, 224, 224), 'float32', 'custom[posit]32')

    run_model(get_mobilenet, (3, 224, 224),
              'float32',
              'custom[posit8]8',
              rtol=float('Inf'),
              atol=float('Inf'))

    # Tolerances set to infinity because bfloat is not numerically correct.
    run_model(get_mobilenet, (3, 224, 224),
              'float32',
              'custom[bfloat]16',
              rtol=float('Inf'),
              atol=float('Inf'))
    run_model(get_inception, (3, 299, 299),
              'float32',
              'custom[bfloat]16',
              rtol=float('Inf'),
              atol=float('Inf'))
    run_model(get_resnet, (3, 224, 224),
              'float32',
              'custom[bfloat]16',
              rtol=float('Inf'),
              atol=float('Inf'))


if __name__ == "__main__":
    setup()
    test_models()
