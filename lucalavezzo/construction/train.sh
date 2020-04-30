#!/bin/bash
source /users/PAS1585/llavez99/work/rl_flow/flow_osuphysics/lucalavezzo/setup_sumo.sh
python -u /users/PAS1585/llavez99/work/rl_flow/flow_osuphysics/lucalavezzo/construction/train_construction_simplified.py >& /users/PAS1585/llavez99/work/rl_flow/logs/training_construction.lg
