import tvm
from tvm import relay
from tvm.relay.testing.mobilenet import get_workload as get_mobilenet
from models.load_mobilenet import load_mobilenet
import numpy as np

module, params, input_shape = load_mobilenet()

# I expected broadcasting to work here, but it didn't
input_shape = (1, ) + input_shape
input = np.random.rand(*input_shape).astype('float32')

ex = relay.create_executor(mod=module)
out = ex.evaluate()(input, **params)
