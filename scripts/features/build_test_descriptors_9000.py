import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


EXTRACT_FEATURES_SCRIPT = Path(__file__).resolve().parent / "extract_features.py"

CLASSES = {
    "Simple_Cracks": "qualitative/images/Simple_Cracks",
    "Multibranched_Crack": "qualitative/images/Multibranched_Crack",
    "Without_Crack": "qualitative/images/Without_Crack",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Construye el set de test extrayendo descriptores D2-Net "
            "sin repetir imagenes del set de train."
        )
    )
    p.add_argument(
        "--train_index",
        default="outputs/all_3_classes_expanded/all_descriptors_index.npz",
        help="Indice global del set train para excluir esas imagenes.",
    )
    p.add_argument("--simple", type=int, default=3000)
    p.add_argument("--multi", type=int, default=3000)
    p.add_argument("--without", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_file", type=str, default="models/d2_tf.pth")
    p.add_argument("--output_extension", type=str, default=".d2-net")
    p.add_argument("--output_type", type=str, default="npz", choices=["npz", "mat"])
    p.add_argument("--preprocessing", type=str, default="caffe", choices=["caffe", "torch"])
    p.add_argument("--max_edge", type=int, default=1600)
    p.add_argument("--max_sum_edges", type=int, default=2800)
    p.add_argument("--multiscale", action="store_true")
    p.add_argument("--out_dir", type=str, default="outputs/all_3_classes_test9000")
    return p.parse_args()


def run_extract_features(list_file, args):
    cmd = [
        sys.executable,
        str(EXTRACT_FEATURES_SCRIPT),
        "--image_list_file",
        str(list_file),
        "--model_file",
        args.model_file,
        "--output_extension",
        args.output_extension,
        "--output_type",
        args.output_type,
        "--preprocessing",
        args.preprocessing,
        "--max_edge",
        str(args.max_edge),
        "--max_sum_edges",
        str(args.max_sum_edges),
    ]
    if args.multiscale:
        cmd.append("--multiscale")
    subprocess.run(cmd, check=True)


def safe_name_from_relpath(rel_path):
    return rel_path.replace("\\", "__").replace("/", "__").replace(".jpg", "")


def load_train_used_images(train_index_path):
    idx = np.load(train_index_path, allow_pickle=True)
    if "image_names" not in idx.files:
        raise ValueError(
            f"{train_index_path} no contiene 'image_names'. "
            "Se espera el indice global generado para train."
        )
    return set(str(x).replace("\\", "/") for x in idx["image_names"])


def sample_without_overlap(folder, n, rng, used_set):
    all_imgs = sorted(Path(folder).glob("*.jpg"))
    all_rel = [str(p).replace("\\", "/") for p in all_imgs]
    candidates = [p for p in all_rel if p not in used_set]
    if n > len(candidates):
        raise ValueError(
            f"Solicitadas {n} imagenes en {folder}, pero solo hay {len(candidates)} "
            "disponibles sin repeticion."
        )
    idx = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in idx]


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    used_train = load_train_used_images(args.train_index)
    counts = {
        "Simple_Cracks": args.simple,
        "Multibranched_Crack": args.multi,
        "Without_Crack": args.without,
    }

    all_parts = []
    image_labels = []
    image_names = []
    image_lengths = []
    class_ranges = {}
    class_start = 0
    train_overlap_count = 0

    for class_name, folder in CLASSES.items():
        n = counts[class_name]
        class_dir = out_root / class_name
        moved_dir = class_dir / "moved_d2net"
        desc_dir = class_dir / "descriptors_only"
        moved_dir.mkdir(parents=True, exist_ok=True)
        desc_dir.mkdir(parents=True, exist_ok=True)

        selected = sample_without_overlap(folder, n, rng, used_train)
        train_overlap_count += len(set(selected).intersection(used_train))

        list_file = class_dir / f"image_list_{class_name}_{n}.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            f.write("\n".join(selected) + "\n")

        run_extract_features(list_file, args)

        class_parts = []
        class_files = []
        class_lengths = []

        for rel in selected:
            src = Path(rel + args.output_extension)
            if not src.exists():
                raise FileNotFoundError(f"No existe salida esperada: {src}")

            safe_stem = safe_name_from_relpath(rel)
            moved_npz = moved_dir / f"{safe_stem}{args.output_extension}"
            if moved_npz.exists():
                moved_npz.unlink()
            shutil.move(str(src), str(moved_npz))

            if args.output_type != "npz":
                raise ValueError("Este script espera output_type=npz para leer 'descriptors'.")

            with np.load(moved_npz) as data:
                desc = data["descriptors"]

            np.save(desc_dir / f"{safe_stem}_descriptors.npy", desc)

            class_parts.append(desc)
            class_files.append(f"{safe_stem}_descriptors.npy")
            class_lengths.append(desc.shape[0])
            image_labels.append(class_name)
            image_names.append(rel)
            image_lengths.append(desc.shape[0])

        class_all = (
            np.concatenate(class_parts, axis=0)
            if class_parts
            else np.empty((0, 512), dtype=np.float32)
        )
        np.save(class_dir / "all_descriptors.npy", class_all)
        np.savez(
            class_dir / "all_descriptors_index.npz",
            files=np.array(class_files),
            lengths=np.array(class_lengths, dtype=np.int64),
            class_name=np.array(class_name),
        )

        all_parts.append(class_all)
        class_end = class_start + class_all.shape[0]
        class_ranges[class_name] = (class_start, class_end)
        class_start = class_end

        print(f"[{class_name}] imagenes={n}, descriptores={class_all.shape[0]}")

    all_descriptors = (
        np.concatenate(all_parts, axis=0)
        if all_parts
        else np.empty((0, 512), dtype=np.float32)
    )
    image_lengths = np.array(image_lengths, dtype=np.int64)
    starts = np.cumsum(np.concatenate(([0], image_lengths[:-1])))
    ends = starts + image_lengths
    image_ranges = np.stack([starts, ends], axis=1)

    np.save(out_root / "all_descriptors.npy", all_descriptors)
    np.savez(
        out_root / "all_descriptors_index.npz",
        image_names=np.array(image_names),
        image_labels=np.array(image_labels),
        image_lengths=image_lengths,
        image_ranges=image_ranges,
        class_names=np.array(list(class_ranges.keys())),
        class_ranges=np.array(list(class_ranges.values()), dtype=np.int64),
    )

    print(f"\nGLOBAL TEST -> descriptors={all_descriptors.shape}")
    print(f"overlap_con_train={train_overlap_count}")
    print(f"Guardado en: {out_root}")


if __name__ == "__main__":
    main()
