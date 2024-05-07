#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -e

. ./getpythonpath.sh

# Install python packages
$PYTHON -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
#$PYTHON -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
$PYTHON -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple wheel
$PYTHON -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.5.1
https_proxy=192.168.149.1:7890 $PYTHON -m pip install -e .
https_proxy=192.168.149.1:7890 $PYTHON -m pip install -e .[dev]
