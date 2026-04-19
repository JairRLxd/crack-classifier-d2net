# Proyecto de deteccion de grietas con D2-Net

Pipeline de clasificacion de grietas en tres clases (Simple_Cracks, Multibranched_Crack, Without_Crack) usando descriptores D2-Net, cuantizacion Bag of Visual Words (BoVW) con KMeans, y clasificadores SVM / Random Forest.

## Estructura del repositorio

```
scripts/
├── preprocessing/        # Realce de imagenes antes de extraer features
│   └── procesar_imagenes_lista.py
├── features/             # Extraccion de descriptores D2-Net y armado del set
│   ├── extract_features.py
│   ├── build_descriptors_dataset.py     # set train
│   └── build_test_descriptors_9000.py   # set test (sin solape con train)
├── bovw/                 # Bag of Visual Words con KMeans
│   ├── kmeansTrain.py    # entrena el codebook y arma histogramas de train
│   └── KMEANSTest.py     # cuantiza el set test con el KMeans entrenado
└── classifiers/          # Clasificadores sobre los histogramas BoVW
    ├── svm/
    │   ├── SVM_train.py      # entrena SVM
    │   ├── SVM_search.py     # GridSearchCV de hiperparametros
    │   └── SVM_test.py       # evaluacion en el set test
    └── rf/
        ├── RF_train.py
        ├── RF_search.py
        └── RF_eval.py

lib/                      # Codigo base de D2-Net (modelo, pyramid, utils)
models/                   # Pesos preentrenados (models/d2_tf.pth)
image_list_*.txt          # Listas de imagenes por clase
outputs/                  # Resultados generados
```

## Dataset

Se utilizo el **Cracks In Concrete Structures (CICS) Dataset**. Del dataset se extrajeron tres clases:

- `Simple_Cracks` — grietas simples
- `Multibranched_Crack` — grietas ramificadas
- `Without_Crack` — imagenes sin grieta

El dataset se separo en dos conjuntos disjuntos (sin imagenes repetidas entre ellos):

- **Train**: 1000 imagenes por clase (3000 totales). A este conjunto se le aplico **mejoramiento de imagenes** (ajuste de contraste/brillo y CLAHE) antes de la extraccion de descriptores. Ver [scripts/preprocessing/](scripts/preprocessing/).
- **Test**: 3000 imagenes por clase (9000 totales). Se usaron **tal cual**, sin preprocesamiento, para evaluar el comportamiento del pipeline sobre imagenes crudas.

Las listas de imagenes usadas en cada conjunto estan en los archivos `image_list_*_1000.txt` (train) y `image_list_*_3000.txt` (test).

## Requisitos

- Python 3.8.20
- CUDA Toolkit 10.0.130 (dependencia de sistema si se usa GPU)
- Dependencias de Python: ver [requirements.txt](requirements.txt)

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Coloca el dataset en una carpeta `images/` con subcarpetas por clase (`images/Simple_Cracks/`, `images/Multibranched_Crack/`, `images/Without_Crack/`). El set qualitativo usado para test va en `qualitative/images/<clase>/`.

## Flujo completo

Todos los comandos se ejecutan desde la raiz del proyecto.

**1. Preprocesar imagenes de train (mejora el contraste con CLAHE)**

```bash
python scripts/preprocessing/procesar_imagenes_lista.py
```

**2. Construir el set de train (muestrea imagenes, extrae descriptores D2-Net y genera el indice global)**

```bash
python scripts/features/build_descriptors_dataset.py \
    --list_simple image_list_Simple_Cracks_1000.txt \
    --list_multi image_list_Multibranched_Crack_1000.txt \
    --list_without image_list_Without_Crack_1000.txt
```

**3. Construir el set de test (sin solape con train)**

```bash
python scripts/features/build_test_descriptors_9000.py
```

**4. Bag of Visual Words**

```bash
python scripts/bovw/kmeansTrain.py --k 800
python scripts/bovw/KMEANSTest.py
```

**5. Entrenar y evaluar clasificadores**

SVM:

```bash
python scripts/classifiers/svm/SVM_train.py
python scripts/classifiers/svm/SVM_test.py
```

Random Forest:

```bash
python scripts/classifiers/rf/RF_train.py
python scripts/classifiers/rf/RF_eval.py
```

Busqueda de hiperparametros (opcional):

```bash
python scripts/classifiers/svm/SVM_search.py
python scripts/classifiers/rf/RF_search.py
```

## Notas

- Las rutas por defecto son relativas (`outputs/...`, `images/...`). Siempre corre los scripts desde la raiz del proyecto.
- `outputs/` esta en `.gitignore` porque contiene artefactos pesados que se regeneran al correr la pipeline.
- Las listas `image_list_*.txt` usan rutas relativas al dataset local.

## Creditos

Este trabajo se apoya en el codigo base de D2-Net. Se conserva `LICENSE` del proyecto original.
