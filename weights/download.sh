#!/bin/bash

mkdir -p xdnn
cd xdnn
wget https://download.visinf.tu-darmstadt.de/data/2021-neurips-fast-axiomatic-attribution/models/xfixup_resnet50_model_best.pth.tar
cd ..

mkdir -p bcos
cd bcos 
wget -O resnet_50-f46c1a4159.pth https://nextcloud.mpi-klsb.mpg.de/index.php/s/3Yk6p86SBBFYSBN/download
cd ..

