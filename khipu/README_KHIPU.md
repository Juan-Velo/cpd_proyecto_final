# Ejecución en Khipu — pasos

Fuentes: https://docs.khipu.utec.edu.pe/ (acceder-a-khipu, transferir-archivos, particiones, tipos-de-cuentas, software/uso, ejemplos/miniconda, ejemplos/mpi-openmp). Consultado 2026-11-20.
Nunca guardar contraseñas en estos archivos. Ojo: varios intentos de contraseña fallidos bloquean la IP.

```bash
# 0) subir el proyecto (desde tu PC, en la carpeta FinalCPD)
scp -r src tests khipu <usuario>@khipu.utec.edu.pe:~/FinalCPD/
ssh <usuario>@khipu.utec.edu.pe
cd ~/FinalCPD && mkdir -p resultados/logs

# 1) diagnóstico -> devolver check_env.log
bash khipu/check_env.sh 2>&1 | tee check_env.log

# 2) entorno conda (una vez, en el nodo de acceso: único con internet)
bash khipu/setup_env.sh

# 3) validación MPI real (partición debug, 30 min máx.)
sbatch khipu/job_validate.sbatch      # revisar resultados/logs/knn_validate_*.out -> "TODO OK"

# 4) experimentos (partición standard)
sbatch khipu/job_experiments.sbatch
squeue -u $USER

# 5) traer resultados (desde tu PC)
scp -r <usuario>@khipu.utec.edu.pe:~/FinalCPD/resultados ./
```

Puntos a confirmar con `check_env.log` antes del paso 4 (la documentación es inconsistente):
- Particiones: la página de particiones lista `debug, standard, …`; los ejemplos MPI usan `investigacion`/`docencia`. Usamos `debug` y `standard`.
- Si `conda create` falla (sin conda-forge), alternativa: `module load` del python3 + MPI del sistema (`py3-mpi4py/3.1.3` aparece bajo gnu12-openmpi4) y `export MPI_LAUNCH="mpirun -np"`.
- Si `mpiexec` de conda no arranca dentro de Slurm: `sbatch --export=ALL,MPI_LAUNCH="srun -n" ...`.
- Tiempo estimado de `job_experiments`: < 1 h (límite pedido 4 h; cuenta pregrado: 8 h, 32 cores, 98 GB).
