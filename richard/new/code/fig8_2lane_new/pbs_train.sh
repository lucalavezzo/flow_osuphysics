#PBS -N train_fig8
#PBS -l walltime=3:00:00
#PBS -l nodes=1:ppn=40
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

singularity exec /users/PYS1027/chnrhughes/docker/ubuntu_osuphysics_flow_v_noflow.sif \
	/bin/bash /users/PYS1027/chnrhughes/work/flowtest/new/code/fig8_2lane_new/train.sh
