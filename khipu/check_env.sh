#!/bin/bash
# Diagnostico del entorno Khipu (ejecutar en el nodo de acceso, NO computa nada pesado).
#   bash khipu/check_env.sh 2>&1 | tee check_env.log
# y devolver check_env.log.
# Fuente de cada comando: https://docs.khipu.utec.edu.pe/ (particiones, tipos-de-cuentas, software/uso)
echo "=== fecha / host";            date; hostname
echo "=== particiones visibles";    sinfo -O "partition" 2>&1
echo "=== sinfo resumen";           sinfo -s 2>&1
echo "=== nodos standard/debug";    sinfo -N -p standard,debug -o "%N %c %m %t" 2>&1
echo "=== limites de cuenta";       myaccount 2>&1
echo "=== modulos (python/conda/mpi)"
module avail 2>&1 | grep -i -E "python|conda|mpich|openmpi|impi|mpi4py|gnu"
echo "=== modulos cargados";        module list 2>&1
echo "=== python3 del sistema";     which python3; python3 --version 2>&1
python3 -c "import numpy, sklearn; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__)" 2>&1
python3 -c "import mpi4py; print('mpi4py', mpi4py.__version__)" 2>&1
echo "=== miniconda";  module load miniconda/3.0 2>&1 && conda --version 2>&1 && conda env list 2>&1
echo "=== cuota home"; quota -s 2>&1 | head; df -h ~ 2>&1 | tail -1
echo "=== FIN"
