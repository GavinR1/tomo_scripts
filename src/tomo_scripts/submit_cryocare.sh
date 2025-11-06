#!/bin/bash
#SBATCH --partition gpunodes
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 1
#SBATCH --job-name cryocare_workflow
#SBATCH --output cryocare_workflow_output.txt
#SBATCH --error cryocare_workflow_error.txt
#SBATCH --open-mode=append

module load cryocare
set -x

cryoCARE_extract_train_data.py --conf ${1} >> cryocare_prep_output.txt
cryoCARE_train.py --conf ${2} >> cryocare_train_output.txt
cryoCARE_predict.py --conf ${3} >> cryocare_predict_output.txt