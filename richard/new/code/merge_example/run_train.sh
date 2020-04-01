#PBS -N train_merge
#PBS -l walltime=8:00:00
#PBS -l nodes=1:ppn=40
#PBS -l mem=5000MB
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

singularity exec /users/PYS1027/chnrhughes/docker/ubuntu_osuphysics_flow_v_noflow.sif /bin/bash /users/PYS1027/chnrhughes/work/flowtest/new/code/merge_example/merge.sh
