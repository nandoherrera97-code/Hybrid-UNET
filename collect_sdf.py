"""
Copia todos los archivos sdf_NACA*.png encontrados en:
  data/raw/results/motorbike_*/constant/triSurface/sdf_NACA*.png
hacia:
  data/raw/sdf/

Si dos ficheros distintos tienen el mismo nombre se añade el prefijo
del directorio motorbike (ej: motorbike_0009_sdf_NACA_1808_0.0deg.png).

Uso:
    cd Hybrid_UNET
    python collect_sdf.py
"""

import shutil
from pathlib import Path

SRC_ROOT = Path("data/raw/results")
DST_DIR  = Path("data/raw/sdf")

# ---- Buscar todos los ficheros fuente ----
sources = sorted(SRC_ROOT.glob("motorbike_*/constant/triSurface/sdf_NACA*.png"))

if not sources:
    print("No se encontraron imágenes. Verifica que la ruta sea correcta.")
    raise SystemExit(1)

print(f"Encontradas {len(sources)} imágenes.")

# ---- Detectar colisiones de nombre ----
name_count: dict[str, int] = {}
for p in sources:
    name_count[p.name] = name_count.get(p.name, 0) + 1

DST_DIR.mkdir(parents=True, exist_ok=True)

# ---- Copiar ----
copied = skipped = 0
for src in sources:
    if name_count[src.name] > 1:
        # Nombre duplicado → añadir prefijo del directorio motorbike
        motorbike_id = src.parts[src.parts.index("results") + 1]  # ej: motorbike_0009
        dst_name = f"{motorbike_id}_{src.name}"
    else:
        dst_name = src.name

    dst = DST_DIR / dst_name

    if dst.exists():
        print(f"  [OMITIDO] ya existe: {dst.name}")
        skipped += 1
        continue

    shutil.copy2(src, dst)
    print(f"  [OK] {src.parent.parent.parent.name}/{src.name}  →  {dst.name}")
    copied += 1

print(f"\nResumen: {copied} copiadas, {skipped} omitidas.")
print(f"Destino: {DST_DIR.resolve()}")
