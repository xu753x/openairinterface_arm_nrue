#!/bin/bash

echo "Insmod NR_Drv"


NR_DRV=./
lsmod | grep nr_drv >& /dev/null
if [ $? -eq 0 ]; then
    rmmod nr_drv
    rm -rf /dev/nr_cdev0
    rm -rf /dev/nr_cdev1
    rm -rf /dev/nr_sys
fi

insmod ${NR_DRV}/nr_drv.ko
mknod /dev/nr_sys c 200 0
mknod /dev/nr_cdev0 c 201 0
mknod /dev/nr_cdev1 c 202 0
mkdir -p /tmp/saveData/en
mkdir -p /tmp/saveData/de

