#!/bin/sh

# Set TVM_CUSTOM_DATATYPES_EVAL_LOG_DIR to the location where this script should
# should create a new directory of test results. Defaults to './logs'.

basedir="$TVM_CUSTOM_DATATYPES_EVAL_LOG_DIR"
if [ -z "$basedir" ]; then
    basedir="./logs"
fi

# A format similar to ISO 8601, in UTC
logdir="$basedir/$(TZ=UTC date +%Y-%m-%d_%H-%M-%SZ)/"
mkdir -p $logdir

python3 ./tests/test-basic.py > $"$logdir/test-basic.log"
python3 ./tests/test1.py > $"$logdir/test1.log"
