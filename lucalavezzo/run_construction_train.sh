#PBS -N training_myEnv
#PBS -l walltime=8:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=5000MB
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

singularity exec /users/PAS1585/llavez99/work/rl_flow/docker/ubuntu_osuphysics_flow_v4.sif /bin/bash /users/PAS1585/llavez99/work/rl_flow/construction.sh
