import tvm
from tvm import relay
import torchvision
from models.resnet50.load import load_resnet50
import numpy as np
from util.change_dtype import change_dtype, convert_ndarray
from util.run_pretrained_model_test import run_pretrained_model_test
from util.load_datatypes import load_libposit_posit8

# Copied from https://github.com/kuangliu/pytorch-cifar/blob/ab908327d44bf9b1d22cd333a4466e85083d3f21/main.py#L36
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
])
# The repo I used to generate the trained params also uses torchvision's data
# loader, but with train=True.
dataset = torchvision.datasets.CIFAR10('.',
                                       train=False,
                                       download=True,
                                       transform=transform)

module, params, image_shape = load_resnet50()

load_libposit_posit8()

# Change the datatype
conversion_executor = relay.create_executor()
expr, params = change_dtype('float32', 'custom[libposit-posit8]8', module['main'],
                            params, conversion_executor)
module = relay.module.Module.from_expr(expr)

run_pretrained_model_test(module, params, dataset, 'custom[libposit-posit8]8')
