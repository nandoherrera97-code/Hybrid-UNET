"""
Construye CNN_df.pkl a partir de los resultados de OpenFOAM.

Flujo esperado:
    results/
        motorbike_0001/  ← caso OpenFOAM completo
        motorbike_0002/
        ...

Adaptar las funciones de lectura al formato de salida de tus scripts
de post-procesado de OpenFOAM (cuttingPlane, sample, etc.).
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_openfoam_field(case_path, field_name, time_step="500"):
    """
    Placeholder: carga un campo escalar/vectorial desde un caso OpenFOAM.

    Sustituir por la lógica real de lectura (e.g. fluidfoam, VTK, CSV exportado
    desde ParaView, o los archivos de postProcessing/cuttingPlane).

    Returns:
        array 2D de shape (80, 200)
    """
    raise NotImplementedError(
        f"Implementar la lectura del campo '{field_name}' desde {case_path}"
    )


def build_dataset(results_dir, output_path, grid_shape=(80, 200)):
    """
    Recorre todos los casos en results_dir, extrae los campos y construye el DataFrame.

    Args:
        results_dir : Path a la carpeta con los casos OpenFOAM (e.g. 'data/raw')
        output_path : ruta donde guardar el .pkl resultante
        grid_shape  : (H, W) de la malla 2D discretizada
    """
    results_dir = Path(results_dir)
    cases = sorted(results_dir.glob("motorbike_*"))

    records = []
    for case in cases:
        try:
            sdf  = load_openfoam_field(case, "SDF")
            ux   = load_openfoam_field(case, "Ux_discretized")
            uy   = load_openfoam_field(case, "Uy_discretized")
            p    = load_openfoam_field(case, "P_discretized")

            records.append({
                "case":           case.name,
                "SDF":            sdf,
                "Ux_discretized": ux,
                "Uy_discretized": uy,
                "P_discretized":  p,
            })
            print(f"  OK: {case.name}")
        except Exception as e:
            print(f"  ERROR en {case.name}: {e}")

    df = pd.DataFrame(records)
    df.to_pickle(output_path)
    print(f"\nDataset guardado en {output_path}  ({len(df)} casos)")
    return df


def normalize_dataset(df):
    """
    Normaliza los campos entre 0 y 1 y devuelve los parámetros de normalización.

    Returns:
        df_norm    : DataFrame con los arrays normalizados
        norm_params: dict {"Ux": (min, max), "Uy": (min, max), "P": (min, max)}
    """
    df_norm = df.copy()
    norm_params = {}

    for field in ["Ux_discretized", "Uy_discretized", "P_discretized"]:
        all_vals = np.concatenate([arr.flatten() for arr in df[field]])
        mn, mx = all_vals.min(), all_vals.max()
        df_norm[field] = df[field].apply(lambda a: (a - mn) / (mx - mn))
        key = field.split("_")[0]
        norm_params[key] = (mn, mx)

    sdf_vals = np.concatenate([arr.flatten() for arr in df["SDF"]])
    mn, mx = sdf_vals.min(), sdf_vals.max()
    df_norm["SDF"] = df["SDF"].apply(lambda a: (a - mn) / (mx - mn))

    return df_norm, norm_params


if __name__ == "__main__":
    build_dataset(
        results_dir="data/raw",
        output_path="data/processed/CNN_df.pkl",
    )
