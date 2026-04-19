import os
os.environ["LOKY_MAX_CPU_COUNT"] = "16"

import argparse
import joblib
import numpy as np
from sklearn.preprocessing import normalize

DESCRIPTORS_PATH = "outputs/all_3_classes_test9000/all_descriptors.npy"
KMEANS_PATH      = "outputs/KmeansResults/K800T2/kmeans.joblib"
OUT_DIR          = "outputs/KMEANSTest/K800T2"
CHUNK_SIZE       = 100_000   # descriptores por bloque en predict
# ──────────────────────────────────────────────


def resolve_scaler_path(kmeans_path):
    return os.path.join(os.path.dirname(os.path.abspath(kmeans_path)), "scaler.joblib")


# ==========================================================
# CARGAR METADATOS
# ==========================================================

def load_image_metadata(descriptors_path):
    root     = os.path.dirname(os.path.abspath(descriptors_path))
    idx_path = os.path.join(root, "all_descriptors_index.npz")

    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"No se encontró el índice: {idx_path}")

    with np.load(idx_path, allow_pickle=True) as idx:
        image_ranges = idx["image_ranges"].reshape(-1, 2).astype(np.int64)
        image_labels = idx["image_labels"]
        image_names  = idx["image_names"]

    return image_ranges, image_labels, image_names


# ==========================================================
# CONSTRUIR HISTOGRAMAS BoVW
# ==========================================================

def build_histograms(word_labels, image_ranges, k):
    hist = np.zeros((len(image_ranges), k), dtype=np.float32)
    for i, (start, end) in enumerate(image_ranges):
        hist[i] = np.bincount(word_labels[start:end], minlength=k)
    return hist


# ==========================================================
# PREDICCIÓN POR BLOQUES
# ==========================================================

def predict_chunks(kmeans, scaler, descriptors, chunk_size):
    n            = len(descriptors)
    word_labels  = np.empty(n, dtype=np.int32)

    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        chunk = np.array(descriptors[start:end], dtype=np.float32)
        chunk = scaler.transform(chunk).astype(np.float32)
        word_labels[start:end] = kmeans.predict(chunk)
        print(f"  Predicción: {end:,} / {n:,}", end="\r")

    print()
    return word_labels


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="BoVW TEST — predict con KMeans ya entrenado")
    parser.add_argument("--descriptors", default=DESCRIPTORS_PATH)
    parser.add_argument("--kmeans_path", default=KMEANS_PATH)
    parser.add_argument("--out_dir",     default=OUT_DIR)
    parser.add_argument("--chunk_size",  type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Cargar descriptores (mmap = sin cargar todo en RAM) ──
    print("\nCargando descriptores (mmap)...")
    descriptors = np.load(args.descriptors, mmap_mode="r")
    print(f"  Shape: {descriptors.shape}  dtype: {descriptors.dtype}")

    # ── Cargar metadatos ─────────────────────────────────────
    print("\nCargando metadatos...")
    image_ranges, image_labels, image_names = load_image_metadata(args.descriptors)
    print(f"  Imágenes: {len(image_labels)}")

    # ── Cargar KMeans entrenado ──────────────────────────────
    print(f"\nCargando KMeans: {args.kmeans_path}")
    kmeans = joblib.load(args.kmeans_path)
    k      = int(kmeans.n_clusters)
    print(f"  K={k} (heredado del modelo entrenado)")

    scaler_path = resolve_scaler_path(args.kmeans_path)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No se encontró el scaler: {scaler_path}")

    print(f"Cargando scaler: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # ── Predicción por bloques con la misma transformación del train ──
    print(f"\nCuantizando por bloques de {args.chunk_size:,}...")
    word_labels = predict_chunks(kmeans, scaler, descriptors, args.chunk_size)

    # ── Histogramas ──────────────────────────────────────────
    print("Construyendo histogramas BoVW...")
    histograms = build_histograms(word_labels, image_ranges, k)
    print("Normalizando histogramas (L2)...")
    histograms = normalize(histograms, norm="l2")

    # ── Guardar ──────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, "histograms_test.npy"),  histograms)
    np.save(os.path.join(args.out_dir, "word_labels_test.npy"), word_labels)
    np.save(os.path.join(args.out_dir, "image_labels_test.npy"),image_labels)
    np.save(os.path.join(args.out_dir, "image_names_test.npy"), image_names)
    np.save(os.path.join(args.out_dir, "image_ranges_test.npy"),image_ranges)
    np.save(os.path.join(args.out_dir, "all_indices_test.npy"), np.arange(len(image_labels), dtype=np.int64))

    print(f"\nListo. Resultados en: {args.out_dir}")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()
