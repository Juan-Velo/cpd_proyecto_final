#!/bin/bash
# Serie complementaria: p > 16 usando hyper-threading (2 procesos por core fisico).
# Cuenta pregrado: cpu=32 logicas = 16 cores fisicos. NO comparable 1:1 con la serie principal.
#SBATCH -J knn_ht
#SBATCH -p standard
#SBATCH -N 1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH -o resultados/logs/%x_%j.out
source khipu/env.sh
MPI_LAUNCH="mpiexec -bind-to hwthread -n"
OUT=resultados/khipu_ht_${SLURM_JOB_ID}
mkdir -p $OUT
lscpu > $OUT/environment.txt
B3="python src/beta3_mpi_optimizado.py"
for p in 16 24 32; do
  $MPI_LAUNCH $p $B3 --n-train 91968 --n-test 3600 --reps 5 --tag strong_ht --out $OUT/strong_ht.csv
  $MPI_LAUNCH $p $B3 --n-train 22992 --n-test 3600 --reps 5 --tag strong_small_ht --out $OUT/strong_small_ht.csv
  $MPI_LAUNCH $p $B3 --reps 5 --tag digits_ht --out $OUT/digits_ht.csv
done
echo "FIN $(date)"
