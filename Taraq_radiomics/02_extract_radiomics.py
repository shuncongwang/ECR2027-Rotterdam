from pathlib import Path
import pandas as pd
from radiomics import featureextractor

from utils import load_config, project_path


def extract_radiomics(config_path="config.yaml"):
    cfg = load_config(config_path)

    project_root = Path(cfg["project"]["dir"])
    input_xlsx = project_path(cfg, "radiomics_input_xlsx")
    output_xlsx = project_path(cfg, "radiomics_output_xlsx")

    case_col = cfg["columns"]["case"]
    image_col = cfg["radiomics"]["image_column"]
    mask_col = cfg["radiomics"]["mask_column"]
    params_file = cfg["radiomics"].get("params_file")

    df = pd.read_excel(input_xlsx)

    if params_file:
        params_path = project_root / params_file
        extractor = featureextractor.RadiomicsFeatureExtractor(str(params_path))
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()

    results = []

    for _, row in df.iterrows():
        case_id = row[case_col]
        image_rel = row.get(image_col)
        mask_rel = row.get(mask_col)

        if pd.isna(image_rel) or pd.isna(mask_rel):
            print(f"[SKIP] {case_id}: missing {image_col} or {mask_col}")
            continue

        image_path = project_root / Path(str(image_rel))
        mask_path = project_root / Path(str(mask_rel))

        if not image_path.exists():
            print(f"[ERROR] Image not found: {image_path}")
            continue
        if not mask_path.exists():
            print(f"[ERROR] Mask not found: {mask_path}")
            continue

        print(f"Processing: {case_id}")

        try:
            feature_vector = extractor.execute(str(image_path), str(mask_path))
            radiomics_features = {
                k: v for k, v in feature_vector.items()
                if not k.startswith("diagnostics_")
            }

            result_row = {
                case_col: case_id,
                "image_sequence": image_col,
                "image_path": image_path.as_posix(),
                "mask_path": mask_path.as_posix(),
            }
            result_row.update(radiomics_features)
            results.append(result_row)

        except Exception as e:
            print(f"[ERROR] {case_id}: {e}")

    result_df = pd.DataFrame(results)
    result_df.to_excel(output_xlsx, index=False)
    print(f"\nSaved to: {output_xlsx}")
    return result_df


if __name__ == "__main__":
    extract_radiomics()
