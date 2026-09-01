from pathlib import Path
import re
import pandas as pd

from utils import load_config, project_path


def collect_mri_paths(project_root: Path, case_prefix: str):
    """
    Scan all .nii.gz files under project_root and collect MRI paths.

    Example folder structure:

        MR_EGD-0001/
            1_T1/
                NIFTI/
                    T1.nii.gz

            2_T1GD/
                NIFTI/
                    T1GD.nii.gz

            3_T2/
                NIFTI/
                    T2.nii.gz

            4_FLAIR/
                NIFTI/
                    FLAIR.nii.gz

            5_MASK/
                NIFTI/
                    MASK.nii.gz


    Output IDs:

        MR_case = MR_EGD-0001
        case    = EGD-0001
    """

    # ========================================================
    # Resolve project root
    # ========================================================

    project_root = project_root.resolve()

    print("=" * 80)
    print("Scanning MRI files")
    print("=" * 80)

    print(f"Project root: {project_root}")

    # ========================================================
    # Find all .nii.gz files recursively
    # ========================================================

    nii_files = list(
        project_root.glob("**/*.nii.gz")
    )

    print(f"Found {len(nii_files)} .nii.gz files.")
    print()

    records = []

    # ========================================================
    # Process each NIfTI file
    # ========================================================

    for nii_file in nii_files:

        # Relative path from project root
        relative_path = nii_file.relative_to(
            project_root
        )

        parts = relative_path.parts

        # ----------------------------------------------------
        # Find folder beginning with case_prefix
        #
        # Example:
        #
        # MR_EGD-0001
        #
        # if case_prefix = "MR_"
        # ----------------------------------------------------

        case_indices = [
            i
            for i, part in enumerate(parts)
            if part.startswith(case_prefix)
        ]

        # ----------------------------------------------------
        # No case folder found
        # ----------------------------------------------------

        if len(case_indices) == 0:

            print(
                "[WARNING] Cannot find case ID:",
                relative_path
            )

            continue

        # ----------------------------------------------------
        # More than one possible case folder
        # ----------------------------------------------------

        if len(case_indices) > 1:

            print(
                "[WARNING] Multiple case folders found:",
                relative_path
            )

            continue

        # ----------------------------------------------------
        # Case folder
        # ----------------------------------------------------

        case_idx = case_indices[0]

        mr_case_id = parts[case_idx]

        # ----------------------------------------------------
        # Remove prefix to create normal case ID
        #
        # MR_EGD-0001
        # ->
        # EGD-0001
        # ----------------------------------------------------

        if mr_case_id.startswith(case_prefix):

            case_id = mr_case_id[
                len(case_prefix):
            ]

        else:

            case_id = mr_case_id

        # ----------------------------------------------------
        # Make sure sequence folder exists
        # ----------------------------------------------------

        if case_idx + 1 >= len(parts) - 1:

            print(
                "[WARNING] Cannot determine sequence:",
                relative_path
            )

            continue

        # ----------------------------------------------------
        # Sequence folder
        #
        # Examples:
        #
        # 1_T1
        # 2_T1GD
        # 3_T2
        # 4_FLAIR
        # 5_MASK
        # ----------------------------------------------------

        sequence_folder = parts[
            case_idx + 1
        ]

        # ----------------------------------------------------
        # Remove leading number
        #
        # 1_T1    -> T1
        # 2_T1GD  -> T1GD
        # 3_T2    -> T2
        # 4_FLAIR -> FLAIR
        # 5_MASK  -> MASK
        # ----------------------------------------------------

        sequence = re.sub(
            r"^\d+_",
            "",
            sequence_folder
        )

        # ----------------------------------------------------
        # Save record
        # ----------------------------------------------------

        records.append(
            {
                "MR_case": mr_case_id,
                "case": case_id,
                "sequence": sequence,
                "relative_path": relative_path.as_posix(),
            }
        )

    # ========================================================
    # Convert to DataFrame
    # ========================================================

    df_long = pd.DataFrame(records)

    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    if df_long.empty:

        print(
            "[WARNING] No valid MRI records found."
        )

        return df_long, df_long

    # ========================================================
    # Sort long-format table
    # ========================================================

    df_long = df_long.sort_values(
        by=[
            "MR_case",
            "case",
            "sequence"
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # Check duplicate sequence files
    # ========================================================

    duplicated = (
        df_long
        .groupby(
            [
                "MR_case",
                "case",
                "sequence"
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    duplicated = duplicated[
        duplicated["count"] > 1
    ]

    if not duplicated.empty:

        print()
        print("=" * 80)

        print(
            "[WARNING] Multiple nii.gz files found "
            "for the same case and sequence:"
        )

        print("=" * 80)

        print(
            duplicated.to_string(
                index=False
            )
        )

        print()

    # ========================================================
    # Convert long format to wide format
    # ========================================================
    #
    # Result:
    #
    # MR_case | case | FLAIR | MASK | T1 | T1GD | T2
    #
    # ========================================================

    df_wide = df_long.pivot_table(
        index=[
            "MR_case",
            "case"
        ],

        columns="sequence",

        values="relative_path",

        # If more than one file exists,
        # join the paths using ;
        aggfunc=lambda x: ";".join(x),
    ).reset_index()

    # Remove pandas column-index name
    df_wide.columns.name = None

    # ========================================================
    # Preferred sequence column order
    # ========================================================

    preferred_sequences = [
        "FLAIR",
        "MASK",
        "T1",
        "T1GD",
        "T2",
    ]

    # Existing MRI sequence columns
    existing_sequences = [
        col
        for col in preferred_sequences
        if col in df_wide.columns
    ]

    # Any additional unexpected sequences
    extra_sequences = [
        col
        for col in df_wide.columns
        if col not in (
            [
                "MR_case",
                "case"
            ]
            +
            preferred_sequences
        )
    ]

    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    final_columns = (
        [
            "MR_case",
            "case"
        ]
        +
        existing_sequences
        +
        extra_sequences
    )

    df_wide = df_wide[
        final_columns
    ]

    # ========================================================
    # Sort wide table
    # ========================================================

    df_wide = df_wide.sort_values(
        by="case"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # Summary
    # ========================================================

    print("=" * 80)
    print("MRI collection summary")
    print("=" * 80)

    print(
        f"Number of MRI files : "
        f"{len(df_long)}"
    )

    print(
        f"Number of cases     : "
        f"{df_wide['case'].nunique()}"
    )

    print()

    print(
        "Sequences found:"
    )

    sequence_counts = (
        df_long["sequence"]
        .value_counts()
        .sort_index()
    )

    print(
        sequence_counts.to_string()
    )

    print("=" * 80)

    return df_long, df_wide


def main(
    config_path="config.yaml"
):

    # ========================================================
    # Load config
    # ========================================================

    cfg = load_config(
        config_path
    )

    # ========================================================
    # Project root
    # ========================================================

    project_root = Path(
        cfg["project"]["dir"]
    )

    # ========================================================
    # Output Excel path
    # ========================================================

    output_xlsx = project_path(
        cfg,
        "mri_paths_xlsx"
    )

    # ========================================================
    # Case prefix
    #
    # Default:
    #
    # MR_
    #
    # Example:
    #
    # MR_EGD-0001
    # ========================================================

    case_prefix = (
        cfg["radiomics"]
        .get(
            "sequence_case_prefix",
            "MR_"
        )
    )

    print()
    print(
        f"Case prefix: {case_prefix}"
    )

    # ========================================================
    # Collect MRI paths
    # ========================================================

    df_long, df_wide = (
        collect_mri_paths(
            project_root,
            case_prefix
        )
    )

    # ========================================================
    # Save Excel
    # ========================================================

    print()
    print("=" * 80)
    print("Saving Excel")
    print("=" * 80)

    print(
        f"Output file: {output_xlsx}"
    )

    with pd.ExcelWriter(
        output_xlsx,
        engine="openpyxl"
    ) as writer:

        # ----------------------------------------------------
        # Wide format
        # ----------------------------------------------------

        df_wide.to_excel(
            writer,
            sheet_name="Paths",
            index=False
        )

        # ----------------------------------------------------
        # Long format
        # ----------------------------------------------------

        df_long.to_excel(
            writer,
            sheet_name="Long",
            index=False
        )

    print()
    print(
        f"Saved to: {output_xlsx}"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()