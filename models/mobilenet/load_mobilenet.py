import tensorflow as tf
import tvm
from tvm import relay
from tvm.relay.testing.mobilenet import get_workload as get_mobilenet


def get_tf_param_name(relay_param_name):
    """Convert relay param name into tf param name"""

    if relay_param_name == 'conv_block_1_conv_weight':
        return 'MobilenetV1/Conv2d_0/weights'
    elif relay_param_name == 'fc_weight':
        return 'MobilenetV1/Logits/Conv2d_1c_1x1/weights'
    else:
        import re
        match = re.match('^separable_conv_block_(\d+)_(\w)(_conv1)?_weight$', relay_param_name)
        if not match:
            raise ValueError(
                'Unexpected variable name: {}'.format(relay_param_name))
        if match.group(1) == 'depthwise':
            return 'MobilenetV1/Conv2d_{}_depthwise/depthwise_weights'.format(
                int(match.group(0)))
        elif match.group(1) == 'conv2':
            return 'MobilenetV1/Conv2d_{}_pointwise/weights'.format(
                int(match.group(0)))

        raise ValueError(
            'Unexpected variable name: {}'.format(relay_param_name))


module, params = get_mobilenet()
module = tvm.relay.transform.SimplifyInference()(module)
module = tvm.relay.transform.DeadCodeElimination()(module)
print(module)
free_vars = tvm.relay.analysis.free_vars(module['main'].body)
free_vars = free_vars[1:]

for free_var in free_vars:
    print(get_tf_param_name(free_var.name_hint))
