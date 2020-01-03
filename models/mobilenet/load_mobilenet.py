import tensorflow as tf
import tvm
from tvm import relay
from tvm.relay.testing.mobilenet import get_workload as get_mobilenet
from tensorflow.python.training import py_checkpoint_reader


vars = tf.train.list_variables('./mobilenet_v1_1.0_224.ckpt')
for var in vars:
    print(var)

def get_tf_param_name(relay_param_name):
    """Convert relay param name into tf param name"""

    # Do a few by hand
    if relay_param_name == 'conv_block_1_conv_weight':
        return 'MobilenetV1/Conv2d_0/weights'
    if relay_param_name == 'conv_block_1_bn_moving_var':
        return 'MobilenetV1/Conv2d_0/BatchNorm/moving_variance'
    if relay_param_name == 'conv_block_1_bn_gamma':
        return 'MobilenetV1/Conv2d_0/BatchNorm/gamma'
    if relay_param_name == 'conv_block_1_bn_moving_mean':
        return 'MobilenetV1/Conv2d_0/BatchNorm/moving_mean'
    if relay_param_name == 'conv_block_1_bn_beta':
        return 'MobilenetV1/Conv2d_0/BatchNorm/beta'
    if relay_param_name == 'fc_weight':
        return 'MobilenetV1/Logits/Conv2d_1c_1x1/weights'

    import re
    match = re.match('^separable_conv_block_(?P<layernum>\d+)_(?P<op>.*)$', relay_param_name)
    if not match:
        raise ValueError(
            'Unexpected variable name: {}'.format(relay_param_name))

    layer_num = match.group('layernum')

    if match.group('op') == 'depthwise_conv1_weight':
        depth_or_point = 'depthwise'
        op = 'depthwise_weights'
    elif match.group('op') == 'bn1_moving_var':
        depth_or_point = 'depthwise'
        op = 'BatchNorm/moving_variance'
    elif match.group('op') == 'bn1_gamma':
        depth_or_point = 'depthwise'
        op = 'BatchNorm/gamma'
    elif match.group('op') == 'bn1_moving_mean':
        depth_or_point = 'depthwise'
        op = 'BatchNorm/moving_mean'
    elif match.group('op') == 'bn1_beta':
        depth_or_point = 'depthwise'
        op = 'BatchNorm/beta'
    elif match.group('op') == 'conv2_weight':
        depth_or_point = 'pointwise'
        op = 'weights'
    elif match.group('op') == 'bn2_moving_var':
        depth_or_point = 'pointwise'
        op = 'BatchNorm/moving_variance'
    elif match.group('op') == 'bn2_gamma':
        depth_or_point = 'pointwise'
        op = 'BatchNorm/gamma'
    elif match.group('op') == 'bn2_moving_mean':
        depth_or_point = 'pointwise'
        op = 'BatchNorm/moving_mean'
    elif match.group('op') == 'bn2_beta':
        depth_or_point = 'pointwise'
        op = 'BatchNorm/beta'
    else:
        raise ValueError(
            'Unexpected or None value for op: {}'.format(match.group('op')))

    return 'MobilenetV1/Conv2d_{}_{}/{}'.format(layer_num, depth_or_point, op)


module, params = get_mobilenet()
module = tvm.relay.transform.SimplifyInference()(module)
module = tvm.relay.transform.DeadCodeElimination()(module)
free_vars = tvm.relay.analysis.free_vars(module['main'].body)
free_vars = free_vars[1:]

reader = py_checkpoint_reader.NewCheckpointReader('./mobilenet_v1_1.0_224.ckpt')

for free_var in free_vars:
    #print(free_var.name_hint)
    name = get_tf_param_name(free_var.name_hint)
    print(reader.get_tensor(name))

