#!/bin/bash
source /users/PYS1027/chnrhughes/docker/setup_after_start.sh
python -u /users/PYS1027/chnrhughes/work/flowtest/new/code/ring/simple_train.py \
	>& /users/PYS1027/chnrhughes/work/flowtest/new/code/ring/logs/pbs_output.log
