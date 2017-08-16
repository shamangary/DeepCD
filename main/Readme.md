# DeepCD Source Code

Author: Tsun-Yi Yang 楊存毅

This folder presents the source codes I used for DeepCD.

DeepCD project is heavily inspired by pnnet https://github.com/vbalnt/pnnet

## Platform
+ Torch7
+ Matlab

## Dependencies
+ Cuda
+ Cudnn
```
luarocks install matio
```
We use MATLAB to save and analysis some information (ex:DDM).

## Parameter concepts

Read note.txt

## Download the UBC dataset (Brown dataset)

Follow https://github.com/vbalnt/UBC-Phototour-Patches-Torch
```
wget http://www.iis.ee.ic.ac.uk/~vbalnt/notredame-t7.tar.gz
wget http://www.iis.ee.ic.ac.uk/~vbalnt/liberty-t7.tar.gz
wget http://www.iis.ee.ic.ac.uk/~vbalnt/yosemite-t7.tar.gz
```
Put the unzipped t7 files under UBCdataset folder

## Simple training command for DeepCD
```
sh runAllDeepCD.sh
```
