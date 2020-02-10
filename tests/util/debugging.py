from sys import stderr
import numpy as np
import tvm
from tvm import relay
from .change_dtype import change_dtype, convert_ndarray
from time import perf_counter_ns

NUM_INFERENCES = 100


def compare_against_float32(module,
                              params,
                              dataset,
                              dtype,
                              num_inferences=NUM_INFERENCES):
    """Run the same model in a custom datatype and in f32, and compare output."""

    conversion_executor = relay.create_executor()
    converted_expr, converted_params = change_dtype('float32', dtype, module['main'],
                                params, conversion_executor)
    converted_module = relay.module.Module.from_expr(converted_expr)
    converted_ex = relay.create_executor(mod=converted_module)
    converted_model = converted_ex.evaluate()

    model  = relay.create_executor(mod=module).evaluate()

    # Only do a certain number of inferences, to save time
    for image, target_class in list(dataset)[:num_inferences]:
        # Add batch dimension
        image_tvm = np.expand_dims(image.numpy().astype('float32'), axis=0)

        expected = model(image_tvm, **params)

        # Change datatype of input
        image_tvm = convert_ndarray(dtype, image_tvm, conversion_executor)

        with tvm.build_config(disable_vectorize=True):
            start = perf_counter_ns()
            output = converted_model(image_tvm, **converted_params)
            elapsed = perf_counter_ns() - start
            output = output.data

        # convert output
        output = convert_ndarray('float32', output, conversion_executor)
        #print(np.where(np.equal(expected.asnumpy(),output.asnumpy())))
        np.testing.assert_allclose(expected.asnumpy(),output.asnumpy())


def compare_against_float32_randominput(module,
                              params,
                              dtype,
                                        shape,
                              num_inferences=NUM_INFERENCES):
    """Run the same model in a custom datatype and in f32, and compare output."""

    conversion_executor = relay.create_executor()
    converted_expr, converted_params = change_dtype('float32', dtype, module['main'],
                                params, conversion_executor)
    converted_module = relay.module.Module.from_expr(converted_expr)
    converted_ex = relay.create_executor(mod=converted_module)
    converted_model = converted_ex.evaluate()

    model  = relay.create_executor(mod=module).evaluate()

    # Only do a certain number of inferences, to save time
    for i in range(num_inferences):
        x = np.random.rand(*shape).astype('float32')
        y = np.random.rand(*shape).astype('float32')

        print(x)
        print(y)

        expected = model(x,y)

        # Change datatype of input
        x = convert_ndarray(dtype, x, conversion_executor)
        y = convert_ndarray(dtype, y, conversion_executor)

        with tvm.build_config(disable_vectorize=True):
            start = perf_counter_ns()
            output = converted_model(x,y)
            elapsed = perf_counter_ns() - start
            output = output.data

        # convert output
        output = convert_ndarray('float32', output, conversion_executor)
        #print(np.where(np.equal(expected.asnumpy(),output.asnumpy())))
        np.testing.assert_allclose(expected.asnumpy(),output.asnumpy())
