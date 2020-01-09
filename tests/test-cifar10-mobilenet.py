import tvm
from tvm import relay
import torchvision
from models.mobilenet_pytorch.load import load_mobilenet
import numpy as np

# Copied from https://github.com/kuangliu/pytorch-cifar/blob/ab908327d44bf9b1d22cd333a4466e85083d3f21/main.py#L36
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
dataset = torchvision.datasets.CIFAR10('.', train=False, download=True, transform=transform)

module, params, image_shape = load_mobilenet()

ex = relay.create_executor(mod=module)
mobilenet = ex.evaluate()

tested = 0
correct = 0
for image, target_class in dataset:
    # Add batch dimension
    image = np.expand_dims(image.numpy(), axis=0)
    logits = mobilenet(image, **params).asnumpy()
    argmax = np.argmax(logits)

    tested += 1
    if (argmax == target_class): correct += 1

    print('{} of {} correct ({})'.format(correct, tested, (correct/tested*100.0)))
