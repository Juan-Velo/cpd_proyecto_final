#!/bin/bash
# Crea (una sola vez) el entorno conda del proyecto en el nodo de acceso
# (unico nodo con internet segun la politica de uso de Khipu).
#   bash khipu/setup_env.sh
# mpi4py + mpich desde conda-forge: el lanzador mpiexec y la libreria MPI
# quedan dentro del mismo entorno (evita mezclar el mpi4py del sistema,
# compilado con openmpi4, con modulos mpich).
set -e
module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q "^knnmpi "; then
  conda create -y -n knnmpi -c conda-forge python=3.11 numpy scikit-learn mpi4py mpich
fi
conda activate knnmpi
python -c "import numpy, sklearn, mpi4py; from mpi4py import MPI; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'mpi4py', mpi4py.__version__); print(MPI.Get_library_version().splitlines()[0])"
which mpiexec
