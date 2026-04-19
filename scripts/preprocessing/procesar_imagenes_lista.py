from pathlib import Path
import argparse
import cv2


DEFAULT_LISTAS = [
    "image_list_Simple_Cracks_1000.txt",
    "image_list_Without_Crack_1000.txt",
    "image_list_Multibranched_Crack_1000.txt",
]


def mejorar_imagen(
    ruta_entrada,
    ruta_salida,
    alpha=1.05,
    beta=4,
    usar_clahe=True,
    clip_limit=1.5,
    tile_size=8,
):
    img = cv2.imread(str(ruta_entrada))
    if img is None:
        raise ValueError(f"No se pudo abrir la imagen: {ruta_entrada}")

    ajustada = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if usar_clahe:
        lab = cv2.cvtColor(ajustada, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size),
        )
        l_mejorado = clahe.apply(l)
        lab_mejorado = cv2.merge((l_mejorado, a, b))
        resultado = cv2.cvtColor(lab_mejorado, cv2.COLOR_LAB2BGR)
    else:
        resultado = ajustada

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(ruta_salida), resultado):
        raise ValueError(f"No se pudo guardar la imagen: {ruta_salida}")


def clase_desde_ruta(ruta_relativa: str, ruta_abs: Path) -> str:
    rel_parent = Path(ruta_relativa).parent
    if str(rel_parent) and str(rel_parent) != ".":
        return rel_parent.name
    return ruta_abs.parent.name


def procesar_desde_lista(lista, salida_root, alpha, beta, usar_clahe, clip_limit, tile_size):
    procesadas = 0
    faltantes = 0
    errores = 0

    with lista.open("r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue

            in_path = Path(rel)
            if not in_path.is_absolute():
                in_path = Path.cwd() / in_path

            if not in_path.exists():
                faltantes += 1
                continue

            clase = clase_desde_ruta(rel, in_path)
            out_path = salida_root / clase / in_path.name

            try:
                mejorar_imagen(
                    in_path,
                    out_path,
                    alpha=alpha,
                    beta=beta,
                    usar_clahe=usar_clahe,
                    clip_limit=clip_limit,
                    tile_size=tile_size,
                )
                procesadas += 1
            except Exception:
                errores += 1

    return procesadas, faltantes, errores


def main():
    parser = argparse.ArgumentParser(
        description="Procesa imagenes de una o varias listas con ajuste suave y guarda por clase."
    )
    parser.add_argument(
        "--listas",
        nargs="*",
        default=DEFAULT_LISTAS,
        help="Archivos .txt con rutas de imagenes",
    )
    parser.add_argument(
        "--salida",
        default="imagenes procesadas suave2",
        help="Carpeta destino raiz",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.05,
        help="Contraste global suave",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=4,
        help="Brillo global suave",
    )
    parser.add_argument(
        "--clip_limit",
        type=float,
        default=1.5,
        help="Intensidad de CLAHE",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=8,
        help="Tamano del grid de CLAHE",
    )
    parser.add_argument(
        "--sin-clahe",
        action="store_true",
        help="Desactiva CLAHE",
    )
    args = parser.parse_args()

    salida_root = Path(args.salida)
    salida_root.mkdir(parents=True, exist_ok=True)

    procesadas_total = 0
    faltantes_total = 0
    errores_total = 0

    for lista_str in args.listas:
        lista = Path(lista_str)
        if not lista.exists():
            raise FileNotFoundError(f"No existe el archivo de lista: {lista}")

        procesadas, faltantes, errores = procesar_desde_lista(
            lista=lista,
            salida_root=salida_root,
            alpha=args.alpha,
            beta=args.beta,
            usar_clahe=not args.sin_clahe,
            clip_limit=args.clip_limit,
            tile_size=args.tile_size,
        )
        procesadas_total += procesadas
        faltantes_total += faltantes
        errores_total += errores
        print(f"Lista procesada: {lista}")
        print(f"Procesadas: {procesadas}")
        print(f"Faltantes: {faltantes}")
        print(f"Errores: {errores}")

    print(f"Procesadas totales: {procesadas_total}")
    print(f"Faltantes totales: {faltantes_total}")
    print(f"Errores totales: {errores_total}")
    print(f"Salida: {salida_root.resolve()}")


if __name__ == "__main__":
    main()
