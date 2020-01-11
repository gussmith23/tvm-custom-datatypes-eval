import tvm
from tvm import relay
from tvm.relay.testing.resnet import get_workload as get_resnet
import torch
from os.path import join, dirname, abspath
import numpy as np


def load_resnet50():

    image_shape = (3, 32, 32)

    class ParamNotFoundError(ValueError):
        def __init__(self, param_name):
            super().__init__(param_name)
            self.param_name = param_name

        def __repr__(self):
            return super().__repr__() + ": {}".format(self.param_name)

    def get_pytorch_param_name(relay_param_name):
        """Convert relay param name into pytorch param name"""

        # Do a few by hand
        if relay_param_name == 'conv0_weight':
            return 'module.conv1.weight'
        if relay_param_name == 'bn0_gamma':
            return 'module.bn1.weight'
        if relay_param_name == 'bn0_beta':
            return 'module.bn1.bias'
        if relay_param_name == 'bn0_moving_mean':
            return 'module.bn1.running_mean'
        if relay_param_name == 'bn0_moving_var':
            return 'module.bn1.running_var'
        if relay_param_name == 'fc1_weight':
            return 'module.linear.weight'
        if relay_param_name == 'fc1_bias':
            return 'module.linear.bias'

        import re
        match = re.match(
            '^stage(?P<stage_num>\d+)_unit(?P<unit_num>\d+)_(?P<op>[a-z]+\d*)_(?P<var_desc>.*)$',
            relay_param_name)
        if not match:
            raise ParamNotFoundError(relay_param_name)

        # What Relay Resnet calls 'stages', the version of Resnet we trained on
        # calls 'layers'
        pytorch_layer_num = int(match.group('stage_num'))

        # What Relay Resnet calls 'units', our training Resnet has no name for, so
        # we call them units as well.
        # They are also 0-indexed, in the pytorch training Resnet
        pytorch_unit_num = int(match.group('unit_num')) - 1

        op = match.group('op')
        if op == 'bn1': pass
        elif op == 'bn2': pass
        elif op == 'bn2': pass
        elif op == 'bn3': pass
        elif op == 'conv1': pass
        elif op == 'conv2': pass
        elif op == 'conv3': pass
        elif op == 'scconv': op = 'shortcut.0'
        elif op == 'scbn': op = 'shortcut.1'
        else:
            raise ValueError('Unexpected or None value for op: {}'.format(op))

        var_desc = match.group('var_desc')
        if var_desc == 'weight': pass
        if var_desc == 'gamma': var_desc = 'weight'
        if var_desc == 'beta': var_desc = 'bias'
        if var_desc == 'moving_mean': var_desc = 'running_mean'
        if var_desc == 'moving_var': var_desc = 'running_var'

        return 'module.layer{}.{}.{}.{}'.format(pytorch_layer_num,
                                                pytorch_unit_num, op, var_desc)

    model = torch.load(join(dirname(abspath(__file__)), 'ckpt.pth'),
                       map_location=torch.device('cpu'))
    model = model['net']

    module, params = get_resnet(num_classes=10,
                                num_layers=50,
                                image_shape=(3, 32, 32))
    module = tvm.relay.transform.SimplifyInference()(module)
    module = tvm.relay.transform.DeadCodeElimination()(module)

    new_params = {}
    for name, var in params.items():
        shape = tuple(map(int, var.shape))

        pytorch_param_name = get_pytorch_param_name(name)
        trained_param_numpy = model[pytorch_param_name].numpy()
        trained_param_numpy_shape = trained_param_numpy.shape

        assert shape == trained_param_numpy_shape, "Shapes for variable %s (matched with %s) do not match: %s vs. %s" % (
            name, pytorch_param_name, shape, trained_param_numpy_shape)

        new_params[name] = trained_param_numpy

    return module, new_params, image_shape
