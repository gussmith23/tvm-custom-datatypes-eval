from tvm.relay.testing.inception_v3 import get_workload as get_inception
from tvm.relay.testing.resnet import get_workload as get_resnet
from tvm.relay.testing.mobilenet import get_workload as get_mobilenet
from util.run_models import run_model
from util.load_datatypes import load_bfloat, load_posit8, load_posit16, load_posit32
from csv import DictWriter
from sys import stdout

load_bfloat()
load_posit8()
load_posit16()
load_posit32()

fieldnames = ['workload', 'datatype', 'time']
table = []

src_dtype_time, dst_dtype_time = run_model(get_mobilenet, (3, 224, 224),
                                           'float32',
                                           'custom[posit8]8',
                                           rtol=float('Inf'),
                                           atol=float('Inf'))
table.append({
    'workload': 'mobilenet',
    'datatype': 'posit8',
    'time': dst_dtype_time
})

src_dtype_time, dst_dtype_time = run_model(get_inception, (3, 299, 299),
                                           'float32',
                                           'custom[posit8]8',
                                           rtol=float('Inf'),
                                           atol=float('Inf'))
table.append({
    'workload': 'inception',
    'datatype': 'posit8',
    'time': dst_dtype_time
})

src_dtype_time, dst_dtype_time = run_model(get_resnet, (3, 224, 224),
                                           'float32',
                                           'custom[posit8]8',
                                           rtol=float('Inf'),
                                           atol=float('Inf'))
table.append({
    'workload': 'resnet',
    'datatype': 'posit8',
    'time': dst_dtype_time
})

# Tolerance set to infinity because bfloat is not numerically correct.
src_dtype_time, dst_dtype_time = run_model(get_mobilenet, (3, 224, 224),
                                           'float32',
                                           'custom[bfloat]16',
                                           rtol=float('Inf'),
                                           atol=float('Inf'))
table.append({
    'workload': 'mobilenet',
    'datatype': 'bfloat16',
    'time': dst_dtype_time
})

# Tolerance set to infinity because bfloat is not numerically correct.
src_dtype_time, dst_dtype_time = run_model(get_inception, (3, 299, 299),
                                           'float32',
                                           'custom[bfloat]16',
                                           rtol=float('Inf'),
                                           atol=float('Inf'))
table.append({
    'workload': 'inception',
    'datatype': 'bfloat16',
    'time': dst_dtype_time
})

# Tolerance set to infinity because bfloat is not numerically correct.
src_dtype_time, dst_dtype_time = run_model(get_resnet, (3, 224, 224),
                                           'float32',
                                           'custom[bfloat]16',
                                           rtol=float('Inf'),
                                           atol=float('Inf'))
table.append({
    'workload': 'resnet',
    'datatype': 'bfloat16',
    'time': dst_dtype_time
})

writer = DictWriter(stdout, fieldnames)
writer.writeheader()
for row in table:
    writer.writerow(row)
