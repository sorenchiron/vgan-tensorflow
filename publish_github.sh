#!/bin/bash
git rm -r --cached ./checkpoints
git rm -r --cached ./datasets
git rm -r --cached ./docs
git rm -r --cached ./misc
git rm -r --cached ./records
git rm -r --cached ./sample
git rm -r --cached ./test
if [ ! -d pretrained_model ]
then
    mkdir pretrained_model
    cp -r checkpoints/hsimg-pixnet_resnet-batchSz_1-imgSz_128-fltrDim_64-L1-lambdaAB_50.0_50.0-res_google pretrained_model/
fi
git add pretrained_model