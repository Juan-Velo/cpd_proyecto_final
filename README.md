# Proyecto 5: KNN paralelo con MPI

**CS4052 – Computación Paralela y Distribuida · UTEC · 2026-II** · Profesor: José Fiestas, PhD

**Integrantes:** Juan David Velo Poma · Farid Jack Aquino Castro · Josue Nehemias Velo Poma

Paralelización con MPI (`mpi4py`) del clasificador KNN de `knn_digits_sec.py` (`load_digits`), siguiendo el DAG del enunciado: **Scatter** de los datos de entrenamiento, **Bcast** de los datos de prueba, cálculo local de los *k* vecinos y **Gather** de los candidatos con voto por mayoría. Experimentos ejecutados en el clúster **Khipu** (UTEC).

📄 **Informe final:** [`informe/main.pdf`](informe/main.pdf)

## Resultados principales

| Resultado | Valor |
|---|---|
| Corrección | Mismas 360 predicciones que el código original para todo *p*; **129/129** comprobaciones en Khipu |
| Speedup con 16 procesos (`n_tr` = 91 968) | **S = 6.61**, E = 0.41 |
| Rendimiento | 8.8 → **58.3 GFLOP/s** (FLOP = 3d·n_tr·n_te) |
| Comunicación | ≤ **1.1 %** del tiempo total |
| Cantidad óptima de procesos | p\* ≈ √(A/B): ≈ 8 para `load_digits`; 16 (todos los núcleos) para los conjuntos grandes |

## Dónde está cada punto del enunciado

| Apartado | Informe | Código |
|---|---|---|
| a) PRAM y código con `comm.scatter` / `comm.bcast` / `comm.gather` | Secs. 2.1–2.3 | [`src/beta2_mpi_basico.py`](src/beta2_mpi_basico.py) |
| b) 3 versiones beta y validación | Sec. 2.6 | [`src/`](src/), [`tests/validate_knn.py`](tests/validate_knn.py) |
| c) Tiempos, accuracy, normalización y *p* óptimo | Secs. 2.4–2.5, 3.1, 3.4 y 3.6 | [`analysis/figs_informe.py`](analysis/figs_informe.py) |
| d) Speedup y FLOP/s vs *p* | Secs. 3.2–3.3 | [`analysis/figs_informe.py`](analysis/figs_informe.py) |
| e) Escalabilidad con tamaño variable | Sec. 3.5 | [`analysis/figs_informe.py`](analysis/figs_informe.py) |

## Estructura

```
src/
  knn_core.py               funciones comunes: datos, bloques, distancias, top-k, voto
  beta1_secuencial.py       beta 1: secuencial (código original y versión NumPy)
  beta2_mpi_basico.py       beta 2: primera versión MPI (comm.scatter / bcast / gather)
  beta3_mpi_optimizado.py   beta 3: versión final (Scatterv / Bcast / Gather) y medición de tiempos
tests/validate_knn.py       validación: beta1 = beta2 = beta3 = código original
khipu/                      entorno y jobs de Slurm usados en Khipu
analysis/figs_informe.py    figuras y constantes del modelo a partir de los CSV
resultados/                 CSV y logs de Khipu (jobs 64269 validación, 64271 experimentos, 64272 hyper-threading)
informe/                    informe (main.tex, main.pdf) y sus figuras
```

## Cómo ejecutar

Requisitos: Python 3.11, NumPy, scikit-learn, mpi4py y MPI. Para las figuras: pandas, SciPy y matplotlib.

```bash
# Beta 1 (secuencial)
python src/beta1_secuencial.py

# Beta 2 y beta 3 con 4 procesos
mpiexec -n 4 python src/beta2_mpi_basico.py
mpiexec -n 4 python src/beta3_mpi_optimizado.py --n-train 91968 --n-test 3600 --reps 5 --out resultados/strong.csv

# Validación (sin MPI y con MPI real)
python tests/validate_knn.py
mpiexec -n 7 python tests/validate_knn.py --mpi

# Figuras del informe a partir de los resultados
python analysis/figs_informe.py resultados/khipu_64271 informe/figs
```

En Khipu (Slurm):

```bash
bash khipu/setup_env.sh               # entorno conda (una vez, en el nodo de acceso)
sbatch khipu/job_validate.sbatch      # validación con MPI real
sbatch khipu/job_experiments.sbatch   # load_digits, Mediano, Grande, tamaño variable y comparación
sbatch khipu/job_ht.sbatch            # hyper-threading (p = 16, 24, 32)
```

## Datos

- `load_digits`: 1797 imágenes 8×8 (d = 64), `test_size=0.2`, `random_state=42` → n_tr = 1437, n_te = 360, k = 3.
- Conjuntos grandes: copias de `load_digits` con ruido gaussiano (σ = 1), redondeadas a enteros en [0, 16], con semilla fija (`make_dataset` en `src/knn_core.py`). Mediano: n_tr = 22 992; Grande: n_tr = 91 968; n_te = 3600.

## Uso de IA

Se usó Claude (vía Claude Code) como herramienta de apoyo. El detalle está en la Sec. 5.3 del informe.
