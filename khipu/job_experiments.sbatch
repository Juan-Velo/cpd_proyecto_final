#!/bin/bash
# Bateria completa de experimentos (escalabilidad fuerte, debil y comparaciones).
#   sbatch khipu/job_experiments.sbatch                 (todas las series)
#   sbatch --export=ALL,SERIES="strong weak" khipu/job_experiments.sbatch
# Cuenta pregrado: cpu=32 logicas = 16 cores fisicos (ThreadsPerCore=2) -> 16 procesos con nomultithread.
#SBATCH -J knn_exp
#SBATCH -p standard
#SBATCH -N 1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH -o resultados/logs/%x_%j.out
source khipu/env.sh
OUT=resultados/khipu_${SLURM_JOB_ID}
mkdir -p $OUT resultados/logs
SERIES=${SERIES:-"digits strong_small strong weak compare"}
REPS=${REPS:-5}
PS="1 2 4 6 8 12 16"
B3="python src/beta3_mpi_optimizado.py"

# --- metadatos de reproducibilidad
{ date; hostname; echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"; lscpu; free -g;
  python -c "import numpy,sklearn,mpi4py;from mpi4py import MPI;print(numpy.__version__,sklearn.__version__,mpi4py.__version__);print(MPI.Get_library_version())";
  python -c "import numpy; numpy.show_config()"; } > $OUT/environment.txt 2>&1

run() { echo ">>> $*"; $MPI_LAUNCH "$@"; }

for s in $SERIES; do case $s in
  digits)        # load_digits original (1437 x 360): domina la comunicacion
    for p in $PS; do run $p $B3 --reps $REPS --tag digits --out $OUT/digits.csv; done ;;
  strong_small)  # fuerte, n_tr = 16*1437, n_te = 3600
    for p in $PS; do run $p $B3 --n-train 22992 --n-test 3600 --reps $REPS --tag strong_small --out $OUT/strong_small.csv; done ;;
  strong)        # fuerte, n_tr = 64*1437, n_te = 3600
    for p in $PS; do run $p $B3 --n-train 91968 --n-test 3600 --reps $REPS --tag strong --out $OUT/strong.csv; done ;;
  weak)          # debil: n_tr/p = 8*1437 constante, n_te = 3600 (trabajo/proceso constante)
    for p in $PS; do run $p $B3 --n-train $((11496*p)) --n-test 3600 --reps $REPS --tag weak --out $OUT/weak.csv; done ;;
  compare)       # mejoras: variante del parcial y comunicacion pickle vs buffer
    for p in 1 4 8 16; do
      run $p $B3 --n-train 91968 --n-test 3600 --reps 3 --variant replicate --tag cmp_replicate --out $OUT/compare.csv
      run $p $B3 --n-train 91968 --n-test 3600 --reps 3 --comm pickle      --tag cmp_pickle    --out $OUT/compare.csv
    done ;;
esac; done
# referencia secuencial pura (beta1, sin MPI) para S(p) con Ts = mejor secuencial
python src/beta1_secuencial.py --modes vector --n-train 91968 --n-test 3600 --out $OUT/beta1_seq.jsonl
python src/beta1_secuencial.py --modes vector --n-train 22992 --n-test 3600 --out $OUT/beta1_seq.jsonl
python src/beta1_secuencial.py --modes original,vector --out $OUT/beta1_seq.jsonl
echo "FIN $(date)"
