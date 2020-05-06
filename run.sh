#!/bin/bash
curdir=`pwd`/GaussianK-SGD/
#cd $curdir
dnn="${dnn:-googlenet}"
source $curdir/exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
density="${density:-0.001}"
compressor="${compressor:-topk}"
nwpernode=2
nstepsupdate=1
#MPIPATH=/usr/local/openmpi/openmpi-4.0.1
PY=python
GRADSPATH=$curdir/logs

#pip install --upgrade pip
#pip install --user -r $curdir/ahmed/GaussianK-SGD/requirements.txt

mpirun --allow-run-as-root -wdir $curdir --mca orte_base_help_aggregate 0 -x NCCL_SOCKET_IFNAME=enp1s0f1 \
     --oversubscribe -np $nworkers -hostfile $curdir/cluster4 -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    -x NCCL_DEBUG=INFO \
    -mca plm_rsh_args "-p 12345" --mca btl_tcp_if_exclude docker0,ens1f0,ens1f1,enp1s0f1,lo,virbr0,ib0,ib1 \
    $PY $curdir/dist_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --density $density --compressor $compressor --saved-dir $GRADSPATH 
