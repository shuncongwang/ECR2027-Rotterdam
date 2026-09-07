from pathlib import Path
import argparse
import pandas as pd

from utils_dl import load_config, project_path


MISSING_TEXT = {"", "nan", "none", "null", "na", "n/a"}


def make_checked_path(
    path: Path,
    target_col: str,
    suffix: str = "_checked",
) -> Path:
    return path.with_name(
        f"{path.stem}_{target_col}{suffix}{path.suffix}"
    )


def _is_missing_series(series: pd.Series) -> pd.Series:
    """Return True for NaN or common textual missing-value markers."""
    as_text = series.astype(str).str.strip().str.lower()
    return series.isna() | as_text.isin(MISSING_TEXT)


def clean_target_file(
    input_path: Path,
    case_col: str,
    target_col: str,
    image_columns,
    invalid_values,
    drop_missing: bool = True,
    output_suffix: str = "_checked",
):
    df = pd.read_excel(input_path)

    required_cols = [case_col, target_col, *image_columns]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required column(s): {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df[case_col] = df[case_col].astype(str).str.strip()

    # ------------------------------------------------------------
    # 1) Find cases with missing MRI sequence(s)
    # ------------------------------------------------------------
    sequence_missing_mask = pd.Series(False, index=df.index)
    missing_sequences_by_case = []

    for idx, row in df.iterrows():
        missing_sequences = [
            col for col in image_columns
            if pd.isna(row[col])
            or str(row[col]).strip().lower() in MISSING_TEXT
        ]

        if missing_sequences:
            sequence_missing_mask.loc[idx] = True
            missing_sequences_by_case.append(
                (row[case_col], missing_sequences)
            )

    # ------------------------------------------------------------
    # 2) Find cases with invalid/missing target
    # ------------------------------------------------------------
    invalid_norm = {
        str(v).strip().lower()
        for v in invalid_values
    }

    target_text = df[target_col].astype(str).str.strip().str.lower()
    target_invalid_mask = target_text.isin(invalid_norm)

    if drop_missing:
        target_invalid_mask = (
            target_invalid_mask
            | _is_missing_series(df[target_col])
        )

    # Print exactly the two requested groups, once per input file.
    print(f"\n{input_path.name}")

    target_cases = df.loc[target_invalid_mask, case_col].tolist()
    if target_cases:
        print(
            f"1. Cases with missing/invalid {target_col} "
            f"({len(target_cases)}): "
            + ", ".join(map(str, target_cases))
        )
    else:
        print(f"1. Cases with missing/invalid {target_col} (0): None")

    if missing_sequences_by_case:
        sequence_text = "; ".join(
            f"{case}: {','.join(seqs)}"
            for case, seqs in missing_sequences_by_case
        )
        print(
            f"2. Cases with missing sequence "
            f"({len(missing_sequences_by_case)}): "
            f"{sequence_text}"
        )
    else:
        print("2. Cases with missing sequence (0): None")

    # Remove a case if either condition is true.
    remove_mask = target_invalid_mask | sequence_missing_mask
    checked = df.loc[~remove_mask].copy()

    output_path = make_checked_path(
        input_path,
        target_col,
        output_suffix,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checked.to_excel(output_path, index=False)

    return output_path, len(df), int(remove_mask.sum()), len(checked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config_resnet.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    case_col = cfg["columns"]["case"]
    target_col = cfg["columns"]["target"]
    image_columns = cfg["columns"]["image_columns"]

    cleaning_cfg = cfg.get("target_cleaning", {})

    invalid_values = cleaning_cfg.get(
        "invalid_values",
        ["Unknown"],
    )
    drop_missing = bool(
        cleaning_cfg.get("drop_missing", True)
    )
    output_suffix = cleaning_cfg.get(
        "output_suffix",
        "_checked",
    )

    train_path = project_path(
        cfg,
        "train_validation_xlsx",
    )
    test_path = project_path(
        cfg,
        "test_xlsx",
    )

    clean_target_file(
        train_path,
        case_col,
        target_col,
        image_columns,
        invalid_values,
        drop_missing,
        output_suffix,
    )

    clean_target_file(
        test_path,
        case_col,
        target_col,
        image_columns,
        invalid_values,
        drop_missing,
        output_suffix,
    )


if __name__ == "__main__":
    main()
