#!/bin/bash
# Diagnostico de contencion de memoria / ubicacion NUMA (ver src/diag_memoria.py)
#SBATCH -J knn_diag
#SBATCH -p standard
#SBATCH -N 1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH -o resultados/logs/%x_%j.out
source khipu/env.sh
OUT=resultados/diag_${SLURM_JOB_ID}
mkdir -p $OUT
{ hostname; lscpu; numactl -H; echo "SLURM_CPU_BIND=$SLURM_CPU_BIND"; } > $OUT/environment.txt 2>&1
for p in 1 2 4 8 12 16; do
  mpiexec -bind-to core   -n $p python src/diag_memoria.py --tag core   --out $OUT/diag.csv
  mpiexec -bind-to core -map-by socket -n $p python src/diag_memoria.py --tag socket --out $OUT/diag.csv
done
# prueba de la mejora en el experimento real (fuerte, n_tr=91968) con reparto por socket
for p in 2 4 6 8 12 16; do
  mpiexec -bind-to core -map-by socket -n $p python src/beta3_mpi_optimizado.py --n-train 91968 --n-test 3600 \
     --reps 5 --tag strong_socket --out $OUT/strong_socket.csv
done
echo "FIN $(date)"
