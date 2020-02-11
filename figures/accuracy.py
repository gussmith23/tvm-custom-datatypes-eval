'''Generate figure for model accuracy across different datatypes.'''

from os import getenv, listdir
from os.path import isfile, join
from re import match
from collections import namedtuple
import pandas as pd
from statistics import mean

LOG_DIR = getenv('LOG_DIR')


def parse_accuracy_test_filename(filename):
    '''Parse test-<dataset>-<modelname>-<datatype>.log.


    Return None if this pattern isn't found.'''
    ParsedFilename = namedtuple('ParsedFilename',
                                ['dataset', 'model', 'datatype'])

    m = match('test-(?P<dataset>\w+)-(?P<model>\w+)-(?P<datatype>.*).log',
              filename)
    if (m):
        return ParsedFilename(m.group('dataset'), m.group('model'),
                              m.group('datatype'))
    else:
        return None


# Get the list of files
files = [(filename, parse_accuracy_test_filename(filename))
         for filename in listdir(LOG_DIR)]
files = list(filter(lambda t: t[1], files))

# Parse data out of files, calculate model accuracy
table = {}
for filename, parse in files:
    with open(join(LOG_DIR, filename), 'r') as f:

        # Read until we see BEGIN TEST
        line = f.readline()
        while line and line != 'BEGIN TEST\n':
            line = f.readline()

        # Read until we find the header
        line = f.readline()
        while line and line == '\n':
            line = f.readline()

        # Clean up the header
        header = line.strip().split(',')
        line = f.readline()

        # Parse the data
        data = []
        while line and line != '\n':
            data.append(tuple(line.strip().split(',')))
            line = f.readline()

        # Calculate model accuracy
        correct_index = header.index('output correct?')
        model_accuracy = mean(list(map(lambda t: int(t[correct_index]), data)))

        # Calculate mean runtime (in ns)
        # TODO variance?
        inference_time_index = header.index('inference time (ns)')
        mean_inference_time = mean(
            list(map(lambda t: int(t[inference_time_index]), data)))

        # Put it all into a big table
        if parse.dataset not in table:
            table[parse.dataset] = {}
        if parse.model not in table[parse.dataset]:
            table[parse.dataset][parse.model] = {}
        if parse.datatype not in table[parse.dataset][parse.model]:
            table[parse.dataset][parse.model][parse.datatype] = {}
        table[parse.dataset][parse.model][
            parse.datatype]['model accuracy'] = model_accuracy
        table[parse.dataset][parse.model][
            parse.datatype]['mean inference time (ns)'] = mean_inference_time

Datatype = namedtuple('Datatype', ['long_name', 'readable_name'])
datatypes = [
    Datatype('float32', 'native float32'),
    Datatype('customtype-noptype', 'noptype'),
    Datatype('customtype-float32', 'float32'),
    Datatype('customtype-bfloat16', 'bfloat16'),
    Datatype('customtype-posit8', 'universal posit8'),
    Datatype('customtype-posit16', 'universal posit16'),
    Datatype('customtype-posit32', 'universal posit32'),
    Datatype('customtype-libposit-posit8', 'libposit posit8'),
    Datatype('customtype-libposit-posit16', 'libposit posit16'),
    Datatype('customtype-libposit-posit32', 'libposit posit32'),
]

for dataset in ['cifar10']:
    for model in ['mobilenet', 'resnet50']:
        print(pd.DataFrame.from_dict(table[dataset][model]))
