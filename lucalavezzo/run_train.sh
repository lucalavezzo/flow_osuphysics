#PBS -N train_construction
#PBS -l walltime=8:00:00
#PBS -l nodes=1:ppn=15
#PBS -l mem=10GB
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

singularity exec /users/PAS1585/llavez99/work/rl_flow/docker/ubuntu_osuphysics_flow_v4.sif /bin/bash /users/PAS1585/llavez99/work/rl_flow/flow_osuphysics/lucalavezzo/construction/construction.sh
