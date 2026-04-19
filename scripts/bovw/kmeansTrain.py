import argparse
import os

import joblib
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize

# ──────────────────────────────────────────────
DESCRIPTORS_PATH = "outputs/imagesprocessed/all_descriptors.npy"
OUT_DIR          = "outputs/KmeansResults/K800T2"
K                = 800
RANDOM_STATE     = 42
MAX_ITER         = 200
N_INIT           = 10
# ──────────────────────────────────────────────5   


# ==========================================================
# CARGAR DESCRIPTORES  (.npy | .npz | .csv)
# ==========================================================

def load_descriptors(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        print("  Formato detectado: CSV")
        return np.loadtxt(path, delimiter=",", dtype=np.float32)

    if ext == ".npy":
        print("  Formato detectado: NPY")
        return np.load(path).astype(np.float32)

    if ext in {".npz", ".d2-net"}:
        print("  Formato detectado: NPZ")
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.keys())
            key  = "descriptors" if "descriptors" in keys else keys[0]
            print(f"  Clave usada: '{key}'  |  Claves disponibles: {keys}")
            return data[key].astype(np.float32)

    raise ValueError(f"Formato no soportado: '{ext}'. Usa .npy, .npz o .csv")


# ==========================================================
# CARGAR METADATOS  (.npz | CSVs separados)
# ==========================================================

def load_image_metadata(descriptors_path):
    root     = os.path.dirname(os.path.abspath(descriptors_path))
    npz_path = os.path.join(root, "all_descriptors_index.npz")

    # ── Opción A: existe el NPZ con todo junto ──────────────
    if os.path.exists(npz_path):
        print(f"  Metadatos desde NPZ: {npz_path}")
        with np.load(npz_path, allow_pickle=True) as data:
            image_ranges = data["image_ranges"].reshape(-1, 2).astype(np.int64)
            image_labels = data["image_labels"]
            image_names  = data["image_names"]
        return image_ranges, image_labels, image_names

    # ── Opción B: CSVs separados en subcarpeta /index ───────
    index_dir = os.path.join(root, "index")
    ranges_csv = os.path.join(index_dir, "all_descriptors_index_image_ranges.csv")
    labels_csv = os.path.join(index_dir, "all_descriptors_index_image_labels.csv")
    names_csv  = os.path.join(index_dir, "all_descriptors_index_image_names.csv")

    if os.path.exists(ranges_csv):
        print(f"  Metadatos desde CSVs en: {index_dir}")
        image_ranges = np.loadtxt(ranges_csv, delimiter=",", dtype=np.int64).reshape(-1, 2)
        image_labels = np.loadtxt(labels_csv, delimiter=",", dtype=str)
        image_names  = np.loadtxt(names_csv,  delimiter=",", dtype=str)
        return image_ranges, image_labels, image_names

    raise FileNotFoundError(
        f"No se encontraron metadatos.\n"
        f"  Buscado NPZ : {npz_path}\n"
        f"  Buscado CSV : {ranges_csv}"
    )


# ==========================================================
# CONSTRUIR HISTOGRAMAS BoVW
# ==========================================================

def build_histograms(word_labels, image_ranges, k):
    hist = np.zeros((len(image_ranges), k), dtype=np.float32)
    for i, (start, end) in enumerate(image_ranges):
        hist[i] = np.bincount(word_labels[start:end], minlength=k)
    return hist
 


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="BoVW KMeans pipeline (.npy / .npz / .csv)")
    parser.add_argument("--descriptors",  default=DESCRIPTORS_PATH)
    parser.add_argument("--out_dir",      default=OUT_DIR)
    parser.add_argument("--k",            type=int, default=K)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE)
    parser.add_argument("--max_iter",     type=int, default=MAX_ITER)
    parser.add_argument("--n_init",       type=int, default=N_INIT)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Descriptores ────────────────────────────────────────
    print("\nCargando descriptores...")
    descriptors = load_descriptors(args.descriptors)
    print(f"  Shape: {descriptors.shape}  dtype: {descriptors.dtype}")

    print("Estandarizando descriptores por característica...")
    scaler = StandardScaler()
    descriptors = scaler.fit_transform(descriptors).astype(np.float32)

    # ── Metadatos ────────────────────────────────────────────
    print("\nCargando metadatos...")
    image_ranges, image_labels, image_names = load_image_metadata(args.descriptors)
    print(f"  Imágenes: {len(image_labels)}")

    # ── KMeans ───────────────────────────────────────────────
    print(f"\nEntrenando KMeans con K={args.k}...")
    kmeans = KMeans(
        n_clusters=args.k,
        random_state=args.random_state,
        max_iter=args.max_iter,
        n_init=args.n_init,
        verbose=1,
    )
    kmeans.fit(descriptors)

    # ── Cuantización ─────────────────────────────────────────
    print("\nCuantizando descriptores...")
    word_labels = kmeans.predict(descriptors)

    # ── Histogramas ──────────────────────────────────────────
    print("Construyendo histogramas BoVW...")
    histograms = build_histograms(word_labels, image_ranges, args.k)

    print("Normalizando histogramas (L2)...")
    histograms = normalize(histograms, norm="l2")

    # ── Guardar ──────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, "histograms.npy"),   histograms)
    np.save(os.path.join(args.out_dir, "word_labels.npy"),  word_labels)
    np.save(os.path.join(args.out_dir, "image_labels.npy"), image_labels)
    np.save(os.path.join(args.out_dir, "image_names.npy"),  image_names)
    np.save(os.path.join(args.out_dir, "image_ranges.npy"), image_ranges)
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))
    joblib.dump(kmeans, os.path.join(args.out_dir, "kmeans.joblib"))

    print(f"\nProceso terminado. Resultados en: {args.out_dir}")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()
