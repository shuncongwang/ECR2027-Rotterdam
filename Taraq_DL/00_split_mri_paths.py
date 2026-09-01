from pathlib import Path

import pandas as pd

from utils import load_config, project_path


def clean_case_column(df: pd.DataFrame, case_col: str) -> pd.DataFrame:
    """Standardize case IDs for safe merging."""
    df = df.copy()
    df[case_col] = df[case_col].astype(str).str.strip()
    return df


def check_unique_cases(df: pd.DataFrame, case_col: str, name: str) -> None:
    """Raise if duplicated case IDs are present."""
    duplicated = df.loc[df[case_col].duplicated(keep=False), case_col].unique()

    if len(duplicated) > 0:
        raise ValueError(
            f"Duplicated case IDs found in {name}:\n"
            f"{duplicated.tolist()}"
        )


def main(config_path: str = "config.yaml") -> None:
    # ============================================================
    # 1. Load configuration
    # ============================================================
    cfg = load_config(config_path)

    case_col = cfg["columns"]["case"]
    split_col = cfg["columns"]["split"]
    target_col = cfg["columns"].get("target", "target")

    train_label = cfg["split"].get(
        "train_validation_label",
        "train-validation",
    )
    test_label = cfg["split"].get(
        "test_label",
        "test",
    )
    keep_split_column = cfg["split"].get(
        "keep_split_column",
        False,
    )

    paths_sheet = cfg["split"].get(
        "mri_paths_sheet",
        "Paths",
    )
    split_sheet = cfg["split"].get(
        "split_sheet",
        0,
    )

    # ============================================================
    # 2. Resolve paths from YAML
    # ============================================================
    mri_paths_xlsx = project_path(cfg, "mri_paths_xlsx")
    split_xlsx = project_path(cfg, "split_xlsx")
    train_output_xlsx = project_path(cfg, "train_validation_xlsx")
    test_output_xlsx = project_path(cfg, "test_xlsx")

    print("=" * 70)
    print("MRI path train-validation / test split")
    print("=" * 70)

    print(f"MRI paths file : {mri_paths_xlsx}")
    print(f"Split file     : {split_xlsx}")
    print(f"Train-val out  : {train_output_xlsx}")
    print(f"Test out       : {test_output_xlsx}")

    # ============================================================
    # 3. Load MRI path table
    # ============================================================
    df_paths = pd.read_excel(
        mri_paths_xlsx,
        sheet_name=paths_sheet,
    )

    if case_col not in df_paths.columns:
        raise ValueError(
            f"Column '{case_col}' not found in MRI paths Excel.\n"
            f"Available columns: {df_paths.columns.tolist()}"
        )

    df_paths = clean_case_column(
        df_paths,
        case_col,
    )

    check_unique_cases(
        df_paths,
        case_col,
        "MRI path table",
    )

    # ============================================================
    # 4. Load split table
    # ============================================================
    df_split = pd.read_excel(
        split_xlsx,
        sheet_name=split_sheet,
    )

    required_split_columns = [
        case_col,
        split_col,
    ]

    for col in required_split_columns:
        if col not in df_split.columns:
            raise ValueError(
                f"Column '{col}' not found in split Excel.\n"
                f"Available columns: {df_split.columns.tolist()}"
            )

    df_split = clean_case_column(
        df_split,
        case_col,
    )

    df_split[split_col] = (
        df_split[split_col]
        .astype(str)
        .str.strip()
    )

    check_unique_cases(
        df_split,
        case_col,
        "split table",
    )

    # ============================================================
    # 5. Validate case matching
    # ============================================================
    path_cases = set(df_paths[case_col])
    split_cases = set(df_split[case_col])

    missing_in_paths = sorted(
        split_cases - path_cases
    )

    missing_in_split = sorted(
        path_cases - split_cases
    )

    if missing_in_paths:
        print(
            f"\n[WARNING] {len(missing_in_paths)} cases are in the split "
            f"table but not in MRI_paths.xlsx:"
        )
        print(missing_in_paths[:20])

    if missing_in_split:
        print(
            f"\n[WARNING] {len(missing_in_split)} cases are in MRI_paths.xlsx "
            f"but not in the split table:"
        )
        print(missing_in_split[:20])

    # ============================================================
    # 6. Merge MRI paths with split information
    # ============================================================
    split_columns = [
        col
        for col in df_split.columns
        if col != case_col
    ]

    df_merged = df_paths.merge(
        df_split[
            [case_col] + split_columns
        ],
        on=case_col,
        how="inner",
        validate="one_to_one",
    )

    if df_merged.empty:
        raise ValueError(
            "No matching cases were found between MRI_paths.xlsx "
            "and the split Excel."
        )

    # ============================================================
    # 7. Validate split labels
    # ============================================================
    available_labels = sorted(
        df_merged[split_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"\nAvailable split labels: {available_labels}"
    )

    train_df = df_merged.loc[
        df_merged[split_col] == train_label
    ].copy()

    test_df = df_merged.loc[
        df_merged[split_col] == test_label
    ].copy()

    if train_df.empty:
        raise ValueError(
            f"No cases found with "
            f"{split_col} == '{train_label}'."
        )

    if test_df.empty:
        raise ValueError(
            f"No cases found with "
            f"{split_col} == '{test_label}'."
        )

    # ============================================================
    # 8. Optional target checks
    # ============================================================
    if target_col in df_merged.columns:
        print("\nTrain-validation target distribution:")
        print(
            train_df[target_col]
            .value_counts(dropna=False)
            .sort_index()
        )

        print("\nTest target distribution:")
        print(
            test_df[target_col]
            .value_counts(dropna=False)
            .sort_index()
        )
    else:
        print(
            f"\n[WARNING] Target column '{target_col}' was not found "
            "in the merged table."
        )

    # ============================================================
    # 9. Remove split column if requested
    # ============================================================
    if not keep_split_column:
        train_df = train_df.drop(
            columns=[split_col],
            errors="ignore",
        )
        test_df = test_df.drop(
            columns=[split_col],
            errors="ignore",
        )

    # ============================================================
    # 10. Save outputs
    # ============================================================
    train_output_xlsx.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    test_output_xlsx.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_excel(
        train_output_xlsx,
        index=False,
    )

    test_df.to_excel(
        test_output_xlsx,
        index=False,
    )

    # ============================================================
    # 11. Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("Split completed")
    print("=" * 70)

    print(
        f"Train-validation: {len(train_df)} cases"
    )
    print(
        f"Test            : {len(test_df)} cases"
    )

    print(
        f"\nSaved train-validation to:\n"
        f"{train_output_xlsx}"
    )

    print(
        f"\nSaved test to:\n"
        f"{test_output_xlsx}"
    )


if __name__ == "__main__":
    main()
