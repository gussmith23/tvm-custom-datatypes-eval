from sys import stderr
import numpy as np
import tvm
from tvm import relay
from .change_dtype import change_dtype, convert_ndarray
from time import perf_counter_ns


def run_pretrained_model_test(module, params, dataset, dtype):
    conversion_executor = relay.create_executor()
    ex = relay.create_executor(mod=module)
    model = ex.evaluate()

    columns = ['inference time (ns)', 'output class number', 'expected class number', 'output correct?']
    print(','.join(columns))

    tested = 0
    correct = 0
    for image, target_class in dataset:
        # Add batch dimension
        image_tvm = np.expand_dims(image.numpy().astype('float32'), axis=0)

        # Change datatype of input
        image_tvm = convert_ndarray(dtype, image_tvm, conversion_executor)

        with tvm.build_config(disable_vectorize=True):
            start = perf_counter_ns()
            output = model(image_tvm, **params)
            elapsed = perf_counter_ns() - start
            output = output.data

        # convert output
        output = convert_ndarray('float32', output, conversion_executor)

        argmax_tvm = np.argmax(output.asnumpy())

        tested += 1
        if (argmax_tvm == target_class): correct += 1
        print('{} of {} correct ({})'.format(correct, tested,
                                             correct / tested),
              file=stderr)

        data = {
            'inference time (ns)': elapsed,
            'output class number': argmax_tvm,
            'expected class number': target_class,
            'output correct?': int(argmax_tvm == target_class)
        }
        print(','.join([str(data[column]) for column in columns]))

    print('Model accuracy:')
    print(correct / tested)
