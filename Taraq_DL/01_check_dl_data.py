from pathlib import Path
import argparse
import pandas as pd

from utils_dl import load_config, project_path


def make_checked_path(
    path: Path,
    target_col: str,
    suffix: str = "_checked",
) -> Path:
    return path.with_name(
        f"{path.stem}_{target_col}{suffix}{path.suffix}"
    )


def clean_target_file(
    input_path: Path,
    case_col: str,
    target_col: str,
    invalid_values,
    drop_missing: bool = True,
    output_suffix: str = "_checked",
):
    print("\n" + "=" * 72)
    print(f"Reading: {input_path}")
    print("=" * 72)

    df = pd.read_excel(input_path)

    if case_col not in df.columns:
        raise ValueError(
            f"Missing case column '{case_col}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    if target_col not in df.columns:
        raise ValueError(
            f"Missing target column '{target_col}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"Original cases: {len(df)}")

    print("\nTarget distribution BEFORE cleaning:")
    print(df[target_col].value_counts(dropna=False))

    # Normalize invalid labels for case-insensitive matching
    invalid_norm = {
        str(v).strip().lower()
        for v in invalid_values
    }

    target_as_text = df[target_col].astype(str).str.strip().str.lower()

    invalid_mask = target_as_text.isin(invalid_norm)

    if drop_missing:
        missing_mask = df[target_col].isna()

        if df[target_col].dtype == "object":
            missing_mask = (
                missing_mask
                | df[target_col].astype(str).str.strip().eq("")
            )

        invalid_mask = invalid_mask | missing_mask

    removed = df.loc[invalid_mask].copy()
    checked = df.loc[~invalid_mask].copy()

    print(f"\nRemoved cases: {len(removed)}")

    if not removed.empty:
        print("\nRemoved case IDs and target values:")
        for _, row in removed.iterrows():
            print(
                f"  {row[case_col]}: "
                f"{target_col}={row[target_col]!r}"
            )
    else:
        print("\nNo matching invalid target values were found.")

    print(f"\nRemaining cases: {len(checked)}")

    print("\nTarget distribution AFTER cleaning:")
    print(checked[target_col].value_counts(dropna=False))

    output_path = make_checked_path(
        input_path,
        target_col,
        output_suffix,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checked.to_excel(
        output_path,
        index=False,
    )

    print(f"\nSaved checked Excel to:\n{output_path}")

    return output_path, len(df), len(removed), len(checked)


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

    cleaning_cfg = cfg.get(
        "target_cleaning",
        {},
    )

    invalid_values = cleaning_cfg.get(
        "invalid_values",
        ["Unknown"],
    )

    drop_missing = bool(
        cleaning_cfg.get(
            "drop_missing",
            True,
        )
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

    train_result = clean_target_file(
        train_path,
        case_col,
        target_col,
        invalid_values,
        drop_missing,
        output_suffix,
    )

    test_result = clean_target_file(
        test_path,
        case_col,
        target_col,
        invalid_values,
        drop_missing,
        output_suffix,
    )

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    print(
        f"Train-validation: "
        f"{train_result[1]} original, "
        f"{train_result[2]} removed, "
        f"{train_result[3]} remaining"
    )

    print(
        f"Test: "
        f"{test_result[1]} original, "
        f"{test_result[2]} removed, "
        f"{test_result[3]} remaining"
    )

    print(
        f"Total removed: "
        f"{train_result[2] + test_result[2]}"
    )


if __name__ == "__main__":
    main()
