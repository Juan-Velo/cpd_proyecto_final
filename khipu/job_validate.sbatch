#!/bin/bash
# Validacion MPI real (beta1 == beta2 == beta3) con p no divisores de n_train.
#   sbatch khipu/job_validate.sbatch          (desde la raiz del proyecto)
#SBATCH -J knn_validate
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH -o resultados/logs/%x_%j.out
source khipu/env.sh
mkdir -p resultados/logs
for p in 1 2 3 4 5 7 8 16; do
  echo "######## p=$p"
  $MPI_LAUNCH $p python tests/validate_knn.py --mpi
done
python tests/validate_knn.py            # suite local (simulada) en el nodo
