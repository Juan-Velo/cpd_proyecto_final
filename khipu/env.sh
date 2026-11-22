# Fuente comun para los jobs (source khipu/env.sh)
module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate knnmpi
# 1 hilo por proceso MPI: evita que BLAS use varios cores y sesgue el speedup
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONWARNINGS=ignore
# Lanzador MPI (sobrescribible: export MPI_LAUNCH="srun -n" antes de sbatch)
MPI_LAUNCH=${MPI_LAUNCH:-"mpiexec -bind-to core -n"}
# nodos sin GPU: desactivar transportes CUDA de UCX (evita "cuDeviceGetCount failed")
export UCX_TLS=self,sm,tcp UCX_WARN_UNUSED_ENV_VARS=n
