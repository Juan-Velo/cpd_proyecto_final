# Proyecto 5: KNN paralelo con MPI

**CS4052 – Computación Paralela y Distribuida · UTEC · 2026-II** · Profesor: José Fiestas, PhD

Paralelización del clasificador KNN (`knn_digits_sec.py`, dataset `load_digits`) con MPI (`mpi4py`), siguiendo el DAG del enunciado: **Scatter** del conjunto de entrenamiento, **Bcast** de las consultas, selección local de los *k* vecinos en `pardo` y **Gather** de los candidatos con voto por mayoría simple. Todos los experimentos se ejecutaron en el clúster **Khipu** (UTEC).

| Documento | Archivo |
|---|---|
| Informe final (PDF) | [`informe/main.pdf`](informe/main.pdf) |
| Presentación (PDF) | [`presentacion/presentacion.pdf`](presentacion/presentacion.pdf) |

**Integrantes:** Juan David Velo Poma · Farid Jack Aquino Castro · Josue Nehemias Velo Poma

---

## Resultados principales

Medidos en Khipu (nodo `n003`, 2×Xeon Gold 6130, hasta 16 núcleos físicos):

| Resultado | Valor |
|---|---|
| Corrección | Predicciones idénticas bit a bit al código original; **129/129** comprobaciones en Khipu (80 con MPI real) |
| Speedup con 16 procesos (`n_tr` = 91 968) | **S = 6.61**, E = 0.41 |
| Rendimiento algorítmico equivalente | 8.8 → **58.3 GFLOP/s** (FLOP = 3d·n_tr·n_te) |
| Comunicación | ≤ **1.1 %** del tiempo total (92k) |
| Cuello de botella | Contención intra-nodo: con trabajo por proceso constante, el cómputo es **φ(16) = 2.52** veces más lento |
| Modelo teórico normalizado | R² = 0.993 (92k); fuera de muestra, R² ≥ 0.994 en las series grandes |

## Correspondencia con el enunciado

| Apartado | Dónde está |
|---|---|
| a) PRAM y código con `comm.scatter` / `comm.bcast` / `comm.gather` | Informe Sec. 2.3 (Algs. 4–5) · [`src/beta2_mpi_basico.py`](src/beta2_mpi_basico.py) |
| b) Desarrollo en 3 versiones beta y validación | Informe Sec. 2.10 · [`src/`](src/) · [`tests/validate_knn.py`](tests/validate_knn.py) |
| c) Tiempos de ejecución, cómputo y comunicación; accuracy; normalización; *p* óptimo | Informe Secs. 3.2–3.5 y 3.9 |
| d) Speedup y FLOP/s vs. *p* | Informe Sec. 3.6 |
| e) Escalabilidad con tamaño variable | Informe Sec. 3.8 |

## Estructura del repositorio

```
src/
  knn_core.py               núcleo común: datos, partición por bloques, distancias, top-k, voto, FLOP
  beta1_secuencial.py       beta 1: secuencial (bucle original, bucle estable o NumPy vectorizado)
  beta2_mpi_basico.py       beta 2: primera versión MPI del DAG (comm.scatter / bcast / gather)
  beta3_mpi_optimizado.py   beta 3: buffers (Scatterv / Bcast / Gather), medición de tiempos, CSV
  diag_memoria.py           diagnóstico de contención intra-nodo (sin comunicación de datos)
tests/
  validate_knn.py           validación: beta1 = beta2 = beta3 = código original (y sklearn)
khipu/                      scripts Slurm y de entorno para Khipu (ver khipu/README_KHIPU.md)
analysis/
  analyze.py                tablas, ajuste del modelo teórico y p* a partir de los CSV
  figs_paper.py             figuras del informe y de la presentación
  validacion_cruzada.py     validación fuera de muestra del modelo extendido
resultados/                 CSV, logs y metadatos de las corridas en Khipu (jobs 64269–64273)
informe/                    informe en LaTeX (main.tex), PDF y figuras
presentacion/               diapositivas en Beamer (presentacion.tex) y PDF
```

## Requisitos

- Python 3.11, NumPy, scikit-learn, mpi4py y una implementación de MPI (en Khipu: MPICH vía conda-forge, ver [`khipu/setup_env.sh`](khipu/setup_env.sh)).
- Para el análisis y las figuras: pandas, SciPy y matplotlib.

## Cómo ejecutar

Desde la raíz del repositorio:

```bash
# Beta 1 (secuencial): modos original, loop y vector sobre load_digits
python src/beta1_secuencial.py

# Beta 2 y beta 3 con MPI (p = 4)
mpiexec -n 4 python src/beta2_mpi_basico.py
mpiexec -n 4 python src/beta3_mpi_optimizado.py --n-train 91968 --n-test 3600 --reps 5 --out resultados/strong.csv

# Validación: partición simulada (sin MPI) y MPI real
python tests/validate_knn.py
mpiexec -n 7 python tests/validate_knn.py --mpi
```

En Khipu (Slurm), el flujo completo está en [`khipu/README_KHIPU.md`](khipu/README_KHIPU.md):

```bash
bash khipu/setup_env.sh                 # entorno conda (una vez, en el nodo de acceso)
sbatch khipu/job_validate.sbatch        # validación con MPI real
sbatch khipu/job_experiments.sbatch     # escalabilidad fuerte, débil y comparaciones
```

Análisis y figuras a partir de los resultados:

```bash
python analysis/analyze.py resultados/khipu_64271
python analysis/figs_paper.py resultados/khipu_64271 informe/figs
python analysis/validacion_cruzada.py resultados/khipu_64271
```

## Datos

- `load_digits`: 1797 imágenes 8×8 (d = 64), con `test_size=0.2` y `random_state=42`, igual que el código base: n_tr = 1437, n_te = 360, k = 3.
- Las series grandes se generan con `make_dataset` (`src/knn_core.py`): copias estratificadas con ruido N(0, 1), redondeadas a enteros en [0, 16] y con semilla 2026. El entrenamiento se genera solo desde X_tr y la prueba solo desde X_te, sin fuga entre ambos.

## Uso de herramientas de IA

Se declara el uso de Claude (vía Claude Code) como herramienta de apoyo. El detalle por ámbito, junto con cómo se verificó cada aporte, está en la Sec. 5.3 del informe (Tabla 7).
