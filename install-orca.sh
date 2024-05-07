#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo 'Installing the Plotly orca dependency for plotly figure export'

SUDO=''
if (( $EUID != 0 )); then
    SUDO='sudo -E'
fi


xargs -a apt-requirements-orca.txt $SUDO apt-get install -y
#the first is author,the second is mine
#export https_proxy=192.168.149.1:7890
#export http_proxy=192.168.149.1:7890
#$SUDO npm install -g --unsafe-perm=true --allow-root electron@6.1.4 orca
#$SUDO npm --registry http://registry.npm.taobao.org install -g --unsafe-perm=true --allow-root npm
export ELECTRON_MIRROR=https://mirrors.huaweicloud.com/electron/
$SUDO npm config set registry http://registry.npmmirror.com
$SUDO npm install -g --unsafe-perm=true --allow-root electron@6.1.4 orca
