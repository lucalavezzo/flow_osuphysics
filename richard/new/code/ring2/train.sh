#!/bin/bash
source /users/PYS1027/chnrhughes/docker/setup_after_start.sh
python -u /users/PYS1027/chnrhughes/work/flowtest/new/code/ring2/simple_train.py \
	>& /users/PYS1027/chnrhughes/work/flowtest/new/code/ring2/logs/pbs_output.log
