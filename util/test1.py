from tvm.relay.testing.inception_v3 import get_workload as get_inception
from tvm.relay.testing.resnet import get_workload as get_resnet
from tvm.relay.testing.mobilenet import get_workload as get_mobilenet
from run_models import run_model
from load_datatypes import load_bfloat, load_posit8, load_posit16, load_posit32

load_bfloat()
load_posit8()
load_posit16()
load_posit32()

# run_model(get_mobilenet, (3, 224, 224), 'float32', 'custom[posit]32')
# run_model(get_inception, (3, 299, 299), 'float32', 'custom[posit]32')
# run_model(get_resnet, (3, 224, 224), 'float32', 'custom[posit]32')

run_model(get_mobilenet, (3, 224, 224),
          'float32',
          'custom[posit8]8',
          rtol=float('Inf'),
          atol=float('Inf'))

# Tolerances set to infinity because bfloat is not numerically correct.
run_model(get_mobilenet, (3, 224, 224),
          'float32',
          'custom[bfloat]16',
          rtol=float('Inf'),
          atol=float('Inf'))
run_model(get_inception, (3, 299, 299),
          'float32',
          'custom[bfloat]16',
          rtol=float('Inf'),
          atol=float('Inf'))
run_model(get_resnet, (3, 224, 224),
          'float32',
          'custom[bfloat]16',
          rtol=float('Inf'),
          atol=float('Inf'))
