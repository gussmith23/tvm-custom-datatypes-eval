import torch
import re
import tvm
from tvm import relay
from tvm.relay.testing.mobilenet import get_workload as get_mobilenet
from os.path import abspath, dirname, join


def load_mobilenet():

    image_shape = (3, 32, 32)

    def get_pytorch_param_name(relay_param_name):
        """Convert relay param name into pytorch param name"""
        """
        module.conv1.weight: torch.Size([32, 3, 3, 3])
        module.bn1.weight: torch.Size([32])
        module.bn1.bias: torch.Size([32])
        module.bn1.running_mean: torch.Size([32])
        module.bn1.running_var: torch.Size([32])
        module.bn1.num_batches_tracked: torch.Size([])
        module.linear.weight: torch.Size([10, 1024])
        module.linear.bias: torch.Size([10])


        conv_block_1_conv_weight: (32, 3, 3, 3)
        conv_block_1_bn_gamma: (32,)
        conv_block_1_bn_beta: (32,)
        conv_block_1_bn_moving_mean: (32,)
        conv_block_1_bn_moving_var: (32,)
        """

        # Do a few by hand
        if relay_param_name == 'conv_block_1_conv_weight':
            return 'module.conv1.weight'
        if relay_param_name == 'conv_block_1_bn_moving_var':
            return 'module.bn1.running_var'
        if relay_param_name == 'conv_block_1_bn_gamma':
            return 'module.bn1.weight'
        if relay_param_name == 'conv_block_1_bn_moving_mean':
            return 'module.bn1.running_mean'
        if relay_param_name == 'conv_block_1_bn_beta':
            return 'module.bn1.bias'
        if relay_param_name == 'fc_weight':
            return 'module.linear.weight'
        if relay_param_name == 'fc_bias':
            return 'module.linear.bias'

        import re
        match = re.match('^separable_conv_block_(?P<layer_num>\d+)_(?P<op>.*)$',
                        relay_param_name)
        if not match:
            raise ValueError(
                'Unexpected variable name: {}'.format(relay_param_name))

        # Pytorch's mobilenet layers are 0-indexed
        layer_num = int(match.group('layer_num')) - 1

        if match.group('op') == 'depthwise_conv1_weight':
            op = 'conv1.weight'
        elif match.group('op') == 'bn1_moving_var':
            op = 'bn1.running_var'
        elif match.group('op') == 'bn1_gamma':
            op = 'bn1.weight'
        elif match.group('op') == 'bn1_moving_mean':
            op = 'bn1.running_mean'
        elif match.group('op') == 'bn1_beta':
            op = 'bn1.bias'
        elif match.group('op') == 'conv2_weight':
            op = 'conv2.weight'
        elif match.group('op') == 'bn2_moving_var':
            op = 'bn2.running_var'
        elif match.group('op') == 'bn2_gamma':
            op = 'bn2.weight'
        elif match.group('op') == 'bn2_moving_mean':
            op = 'bn2.running_mean'
        elif match.group('op') == 'bn2_beta':
            op = 'bn2.bias'
        else:
            raise ValueError('Unexpected or None value for op: {}'.format(
                match.group('op')))

        return 'module.layers.{}.{}'.format(layer_num, op)


    model = torch.load(join(dirname(abspath(__file__)), 'ckpt.pth'), map_location=torch.device('cpu'))

    # Make sure to correctly set the layout, image shape, num classes
    module, params = get_mobilenet(layout='NCHW',
                                    image_shape=(3, 32, 32),
                                    num_classes=10)
    module = tvm.relay.transform.SimplifyInference()(module)
    module = tvm.relay.transform.DeadCodeElimination()(module)
    free_vars = tvm.relay.analysis.free_vars(module['main'].body)
    # Skip data
    assert free_vars[0].name_hint == 'data'
    free_vars = free_vars[1:]

    # Maps strings to NDArrays, like the old `params` object, but the NDArrays are
    # the trained parameters.
    new_params = {}

    for name, var in params.items():
        shape = tuple(map(int, var.shape))
        pytorch_param_name = get_pytorch_param_name(name)
        trained_param_numpy = model['net'][pytorch_param_name].numpy()
        trained_param_numpy_shape = trained_param_numpy.shape

        assert shape == trained_param_numpy_shape, "Shapes for variable %s do not match: %s vs. %s" % (
            name, shape, trained_param_numpy_shape)

        new_params[name] = trained_param_numpy

    return module, new_params, image_shape
