from sys import stderr
import numpy as np
import tvm
from tvm import relay
from .change_dtype import change_dtype, convert_ndarray


def run_pretrained_model_test(module, params, dataset, dtype):
    conversion_executor = relay.create_executor()
    ex = relay.create_executor(mod=module)
    model = ex.evaluate()

    tested = 0
    correct = 0
    for image, target_class in dataset:
        # Add batch dimension
        image_tvm = np.expand_dims(image.numpy().astype('float32'), axis=0)

        # Change datatype of input
        image_tvm = convert_ndarray(dtype, image_tvm, conversion_executor)

        with tvm.build_config(disable_vectorize=True):
            output = model(image_tvm, **params)
            output = output.data

        # convert output
        output = convert_ndarray('float32', output, conversion_executor)

        argmax_tvm = np.argmax(output.asnumpy())

        tested += 1
        if (argmax_tvm == target_class): correct += 1
        print('{} of {} correct ({})'.format(correct, tested,
                                             correct / tested),
              file=stderr)

    print('Model accuracy:')
    print(correct / tested)
