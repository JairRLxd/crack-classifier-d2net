import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


EXTRACT_FEATURES_SCRIPT = Path(__file__).resolve().parent / "extract_features.py"

CLASSES = {
    "Simple_Cracks": "images/Simple_Cracks",
    "Multibranched_Crack": "images/Multibranched_Crack",
    "Without_Crack": "images/Without_Crack",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Muestrea imagenes por clase, extrae D2-Net y junta descriptores."
    )
    p.add_argument("--simple", type=int, default=1000, help="Cantidad para Simple_Cracks")
    p.add_argument("--multi", type=int, default=1000, help="Cantidad para Multibranched_Crack")
    p.add_argument("--without", type=int, default=1000, help="Cantidad para Without_Crack")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--list_simple",
        type=str,
        default="image_list_Simple_Cracks_1000.txt",
        help="TXT con rutas de Simple_Cracks",
    )
    p.add_argument(
        "--list_multi",
        type=str,
        default="image_list_Multibranched_Crack_1000.txt",
        help="TXT con rutas de Multibranched_Crack",
    )
    p.add_argument(
        "--list_without",
        type=str,
        default="image_list_Without_Crack_1000.txt",
        help="TXT con rutas de Without_Crack",
    )
    p.add_argument(
        "--list_root",
        type=str,
        default="images",
        help="Raiz de imagenes a usar cuando se dan listas (p.ej. imagenes procesadas)",
    )
    p.add_argument("--model_file", type=str, default="models/d2_tf.pth")
    p.add_argument("--output_extension", type=str, default=".d2-net")
    p.add_argument("--output_type", type=str, default="npz", choices=["npz", "mat"])
    p.add_argument("--preprocessing", type=str, default="caffe", choices=["caffe", "torch"])
    p.add_argument("--max_edge", type=int, default=1600)
    p.add_argument("--max_sum_edges", type=int, default=2800)
    p.add_argument("--multiscale", action="store_true")
    p.add_argument("--out_dir", type=str, default="outputs/imagesprocessed")
    return p.parse_args()


def sample_images(folder, n, rng):
    imgs = sorted(Path(folder).glob("*.jpg"))
    if n > len(imgs):
        raise ValueError(f"Solicitadas {n} imagenes en {folder}, pero solo hay {len(imgs)}")
    idx = rng.choice(len(imgs), size=n, replace=False)
    return [imgs[i] for i in idx]


def read_list(list_path, class_name, list_root):
    list_path = Path(list_path)
    if not list_path.exists():
        raise FileNotFoundError(f"No existe el archivo de lista: {list_path}")

    root = Path(list_root) if list_root else None
    if root and not root.exists():
        raise FileNotFoundError(f"No existe la carpeta list_root: {root}")

    selected = []
    missing = 0

    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            if root:
                in_path = root / class_name / Path(rel).name
            else:
                in_path = Path(rel)
                if not in_path.is_absolute():
                    in_path = Path.cwd() / in_path
            if not in_path.exists():
                missing += 1
            selected.append(in_path)

    if missing:
        raise FileNotFoundError(f"{missing} imagenes no existen para {class_name} usando {list_path}")

    return selected


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
    # Evita choques de nombre entre carpetas
    return rel_path.replace("\\", "__").replace("/", "__").replace(".jpg", "")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    list_mode = any([args.list_simple, args.list_multi, args.list_without])
    if list_mode and not all([args.list_simple, args.list_multi, args.list_without]):
        raise ValueError("Si usas listas, debes pasar --list_simple, --list_multi y --list_without.")

    counts = {
        "Simple_Cracks": args.simple,
        "Multibranched_Crack": args.multi,
        "Without_Crack": args.without,
    }
    list_paths = {
        "Simple_Cracks": args.list_simple,
        "Multibranched_Crack": args.list_multi,
        "Without_Crack": args.list_without,
    }

    all_parts = []
    image_labels = []
    image_names = []
    image_lengths = []
    class_ranges = {}
    class_start = 0

    for class_name, folder in CLASSES.items():
        if list_mode:
            selected = read_list(list_paths[class_name], class_name, args.list_root)
            n = len(selected)
        else:
            n = counts[class_name]
            selected = sample_images(folder, n, rng)

        class_dir = out_root / class_name
        moved_dir = class_dir / "moved_d2net"
        desc_dir = class_dir / "descriptors_only"
        moved_dir.mkdir(parents=True, exist_ok=True)
        desc_dir.mkdir(parents=True, exist_ok=True)

        list_file = class_dir / f"image_list_{class_name}_{n}.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for p in selected:
                f.write(str(p).replace("\\", "/") + "\n")

        # 1) Extraer features
        run_extract_features(list_file, args)

        # 2) Mover y extraer descriptores
        class_parts = []
        class_files = []
        class_lengths = []

        for img_path in selected:
            rel = str(img_path).replace("\\", "/")
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
                desc = data["descriptors"]  # (Ni, 512)

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

    # 3) Unir todo global
    all_descriptors = (
        np.concatenate(all_parts, axis=0) if all_parts else np.empty((0, 512), dtype=np.float32)
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
        image_ranges=image_ranges,  # clave: rango descriptor por imagen
        class_names=np.array(list(class_ranges.keys())),
        class_ranges=np.array(list(class_ranges.values()), dtype=np.int64),
    )

    print(f"\nGLOBAL -> descriptors={all_descriptors.shape}")
    print(f"Guardado en: {out_root}")


if __name__ == "__main__":
    main()
