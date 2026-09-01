import pandas as pd

from utils import load_config, project_path


def main(config_path="config.yaml"):
    cfg = load_config(config_path)

    feature_xlsx = project_path(cfg, "radiomics_output_xlsx")
    clinical_xlsx = project_path(cfg, "clinical_xlsx")
    split_xlsx = project_path(cfg, "split_xlsx")
    merged_xlsx = project_path(cfg, "merged_xlsx")
    trainval_xlsx = project_path(cfg, "train_validation_xlsx")
    test_xlsx = project_path(cfg, "test_xlsx")

    case_col = cfg["columns"]["case"]
    split_col = cfg["columns"]["split"]
    train_label = cfg["split"]["train_validation_label"].strip().lower()
    test_label = cfg["split"]["test_label"].strip().lower()
    keep_split = bool(cfg["split"].get("keep_split_column", False))

    df_feature = pd.read_excel(feature_xlsx)
    df_clinical = pd.read_excel(clinical_xlsx)

    for df in (df_feature, df_clinical):
        df[case_col] = df[case_col].astype(str).str.strip()

    df_merged = pd.merge(
        df_clinical,
        df_feature,
        on=case_col,
        how="left",
        validate="one_to_one",
    )
    df_merged.to_excel(merged_xlsx, index=False)
    print(f"Merged file saved to:\n{merged_xlsx}")

    df_split = pd.read_excel(split_xlsx)
    df_split[case_col] = df_split[case_col].astype(str).str.strip()
    df_split[split_col] = df_split[split_col].astype(str).str.strip().str.lower()

    if df_split[case_col].duplicated().any():
        dup = df_split.loc[df_split[case_col].duplicated(), case_col].tolist()
        raise ValueError(f"Duplicated case IDs in split file: {dup[:20]}")

    df_all = pd.merge(
        df_merged,
        df_split[[case_col, split_col]],
        on=case_col,
        how="left",
        validate="one_to_one",
    )

    missing_split = df_all.loc[df_all[split_col].isna(), case_col]
    if len(missing_split) > 0:
        print("\n[WARNING] Cases without split information:")
        for case in missing_split:
            print(case)

    df_trainval = df_all[df_all[split_col] == train_label].copy()
    df_test = df_all[df_all[split_col] == test_label].copy()

    if not keep_split:
        df_trainval = df_trainval.drop(columns=split_col)
        df_test = df_test.drop(columns=split_col)

    df_trainval.to_excel(trainval_xlsx, index=False)
    df_test.to_excel(test_xlsx, index=False)

    print(f"\nTotal cases            : {len(df_all)}")
    print(f"Train-validation cases : {len(df_trainval)}")
    print(f"Test cases             : {len(df_test)}")
    print(f"\nTrain-validation saved to:\n{trainval_xlsx}")
    print(f"\nTest saved to:\n{test_xlsx}")


if __name__ == "__main__":
    main()
