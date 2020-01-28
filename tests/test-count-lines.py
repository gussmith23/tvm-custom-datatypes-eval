from os import path
from glob import glob
import re

dir = path.dirname(path.abspath(__file__))

print('{}, {}'.format('filename', 'number of lines'))

for filename in glob(dir + '/examples-for-line-counting/*.py.txt'):
    with open(filename, 'r') as f:
        lines = 0
        count_lines = False
        for l in f:
            if count_lines: lines += 1
            if re.match(' *# START COUNTING.*', l): count_lines = True
            elif re.match(' *# STOP COUNTING.*', l): count_lines = False
        print('{}, {}'.format(path.basename(filename), lines))
