import tensorflow as tf
import tvm
from tvm import relay
from tvm.relay.testing.mobilenet import get_workload as get_mobilenet
from tensorflow.python.training import py_checkpoint_reader
from os.path import abspath, join, dirname


def load_mobilenet():

    image_shape = (224, 224, 3)

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
        match = re.match('^separable_conv_block_(?P<layernum>\d+)_(?P<op>.*)$',
                         relay_param_name)
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
            raise ValueError('Unexpected or None value for op: {}'.format(
                match.group('op')))

        return 'MobilenetV1/Conv2d_{}_{}/{}'.format(layer_num, depth_or_point,
                                                    op)

    # Make sure to set the layout, image shape, num classes to match TF style
    module, params = get_mobilenet(layout='NHWC',
                                   image_shape=image_shape,
                                   num_classes=1001)
    module = tvm.relay.transform.SimplifyInference()(module)
    module = tvm.relay.transform.DeadCodeElimination()(module)
    free_vars = tvm.relay.analysis.free_vars(module['main'].body)
    # Skip data
    assert free_vars[0].name_hint == 'data'
    free_vars = free_vars[1:]

    reader = py_checkpoint_reader.NewCheckpointReader(
        join(dirname(abspath(__file__)), 'mobilenet_v1_1.0_224.ckpt'))

    # Maps strings to NDArrays, like the old `params` object, but the NDArrays are
    # the trained parameters.
    new_params = {}
    for name, var in params.items():
        shape = tuple(map(int, var.shape))
        tf_tensor_name = get_tf_param_name(name)
        tf_tensor = reader.get_tensor(tf_tensor_name)

        # Hard-coding this transformation.
        if name == 'fc_weight':
            tf_tensor = tf_tensor.squeeze().swapaxes(0, 1)

        tf_tensor_shape = tf_tensor.shape

        assert shape == tf_tensor_shape, "Shapes for variable %s do not match: %s vs. %s" % (
            name, shape, tf_tensor_shape)

        new_params[name] = tf_tensor

    return module, new_params, image_shape
