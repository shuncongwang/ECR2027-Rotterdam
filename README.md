# EGD 3D ResNet / MedicalNet Binary Classification Pipeline

This project trains a 3D ResNet classifier on NIfTI MRI data using the **same fixed train-validation / independent test split** produced by the previous XLSX pipeline.

## What this pipeline does

- Reads `train_validation.xlsx` and `test.xlsx` directly.
- Uses only the configured Case ID, target, image-path columns, and optional mask path; radiomics/clinical columns can remain in the same XLSX and are ignored by the CNN.
- Supports 3D ResNet depths **10, 18, 34, 50, 101, 152, 200**.
- Can initialize from MedicalNet/Med3D checkpoints or train the same architecture from scratch.
- Replaces the original MedicalNet segmentation head with a classification head:
  `layer4 -> AdaptiveAvgPool3d(1) -> Dropout -> Linear(2)`.
- Uses an internal stratified validation split only within `train_validation.xlsx`.
- Selects the best epoch by validation AUC and optionally chooses the classification threshold using Youden's J on validation data only.
- Evaluates the untouched `test.xlsx` once and reports AUC, F1, sensitivity, specificity, accuracy, precision, NPV, and bootstrap 95% CIs.

## MedicalNet compatibility fixes included here

The upstream MedicalNet repository is old and its demo was designed around segmentation/transfer learning with PyTorch 0.4.1 and CUDA 9. This pipeline modernizes the relevant parts instead of copying the original training script verbatim.

Important fixes:

1. **Segmentation head removed.** The original `conv_seg` path is not used. The encoder ends at `layer4`, then adaptive global pooling and a binary classification head are applied.
2. **No fixed input-size pooling.** `AdaptiveAvgPool3d(1)` avoids dependence on MedicalNet's old `sample_input_D/H/W` assumptions.
3. **Depth-specific shortcut configuration.** MedicalNet's published settings use shortcut B for ResNet-10/50 and shortcut A for ResNet-18/34; deeper standard settings use B. This implementation follows those published configurations.
4. **Modern shortcut A.** The old implementation used legacy `Variable`/device-specific operations. Here shortcut A is implemented with device-safe PyTorch tensor operations.
5. **Checkpoint wrappers/prefixes handled.** `state_dict`, `model_state_dict`, and common `module.` prefixes are handled.
6. **Segmentation weights are ignored.** Only encoder tensors whose names and shapes match are loaded. `conv_seg`, `fc`, and classifier heads are skipped.
7. **Input-channel mismatch handled.** MedicalNet checkpoints are normally 1-channel. If multiple MRI sequence columns are configured, the first convolution can be adapted by repeating the pretrained single-channel kernel and dividing by the new channel count.
8. **Problematic checkpoints fail visibly.** The loader writes `pretrained_loading_report.json`. If zero tensors load, training stops instead of silently using random initialization. This matters because community reports have documented checkpoint/model-definition mismatches, especially around some ResNet-18/34 files.
9. **No test-set threshold tuning.** The threshold is determined on validation data and stored with the checkpoint, preventing test leakage.
10. **No silent multimodal misregistration.** If multiple MRI sequences or a mask have different array shapes, the dataset raises an error. Registration/resampling should be done upstream rather than resizing each modality independently and pretending they are aligned.

## Expected XLSX format

Your existing XLSX can contain many additional clinical/radiomics columns. The CNN uses only the configured columns.

Example:

| case | target | image_path | mask_path | Age | rad_01 | ... |
|---|---:|---|---|---:|---:|---|
| MR_EGD-0001 | 0 | MR_EGD-0001/1_T1/NIFTI/T1.nii.gz | MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz | 54 | ... | ... |
| MR_EGD-0002 | 1 | MR_EGD-0002/1_T1/NIFTI/T1.nii.gz | MR_EGD-0002/5_MASK/NIFTI/MASK.nii.gz | 61 | ... | ... |

Paths may be absolute or relative to `project.dir`.

If your previous XLSX uses `image_path` / `mask_path`, the default YAML already matches it. If it instead contains sequence-specific path columns such as `T1`, `T1GD`, `T2`, and `FLAIR`, simply configure:

```yaml
columns:
  image_columns: ["T1", "T1GD", "T2", "FLAIR"]
  mask_column: "MASK"
```

The number of CNN input channels is inferred from `image_columns`.

## Configuration

Edit only `config_resnet.yaml` for most experiments.

Common parameters:

```yaml
model:
  depth: 50
  use_pretrained: true
  dropout: 0.20

training:
  learning_rate: 0.0001
  batch_size: 2
  epochs: 80
  validation_fraction: 0.20

input:
  output_size: [64, 128, 128]
```

For a scratch baseline:

```yaml
model:
  depth: 50
  initialization: "scratch"
  use_pretrained: false
```

## Environment

The environment intentionally uses **Python 3.9** and **NumPy 1.26.4 (NumPy 1.x)**. This is also compatible with the earlier PyRadiomics workflow and avoids the NumPy-2 ABI problem encountered with older radiomics C extensions.

Create the environment:

```powershell
conda env create -f environment_resnet.yml
conda activate egd-resnet
```

### Install PyTorch with GPU support

PyTorch is installed separately because the correct build depends on the NVIDIA driver/CUDA runtime available on your workstation. For a known-stable Python 3.9 setup, one option is PyTorch 2.2.x. Select the CUDA build appropriate for your machine from the official PyTorch installation instructions.

Example CUDA 12.1 conda installation:

```powershell
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

CPU-only example:

```powershell
conda install pytorch==2.2.2 torchvision==0.17.2 cpuonly -c pytorch
```

Verify:

```powershell
python -c "import torch, numpy; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('numpy', numpy.__version__)"
```

Expected NumPy:

```text
1.26.4
```

## MedicalNet pretrained checkpoint

Download a MedicalNet/Med3D checkpoint separately and place it under the project directory, for example:

```text
E:/Marion_data/EGD/pretrain/resnet_50_23dataset.pth
```

Then set:

```yaml
paths:
  pretrained_checkpoint: "pretrain/resnet_50_23dataset.pth"

model:
  depth: 50
  use_pretrained: true
```

The configured depth must match the checkpoint architecture. Do not use a ResNet-50 checkpoint with a ResNet-18 model.

The official MedicalNet 23-dataset release listed these principal settings:

- ResNet-10: shortcut B
- ResNet-18: shortcut A
- ResNet-34: shortcut A
- ResNet-50: shortcut B

Community reports indicate that some distributed ResNet-18/34 checkpoints have state-dict shape compatibility problems. Always inspect `DL_results/pretrained_loading_report.json` after loading. A very low loaded-tensor count is a warning that the checkpoint and architecture do not match.

## Run order

From this folder:

```powershell
conda activate egd-resnet
python 01_check_dl_data.py --config config_resnet.yaml
python 02_train_resnet.py --config config_resnet.yaml
python 03_test_resnet.py --config config_resnet.yaml
```

### 1. Data check

`01_check_dl_data.py` checks:

- required XLSX columns;
- duplicate case IDs;
- train/test overlap;
- binary target encoding;
- existence of every configured NIfTI image/mask path.

### 2. Training

`02_train_resnet.py`:

- stratifies `train_validation.xlsx` into training and internal validation subsets;
- loads optional MedicalNet encoder weights;
- trains with weighted cross-entropy when requested;
- applies early stopping on validation AUC;
- stores the best validation-derived classification threshold;
- saves the best checkpoint and training history.

Outputs include:

```text
DL_results/
  best_resnet_classifier.pt
  training_history.xlsx
  validation_predictions.xlsx
  pretrained_loading_report.json   # when MedicalNet weights are used
```

### 3. Independent test

`03_test_resnet.py` uses only the saved checkpoint/validation threshold and evaluates the untouched test set.

Outputs:

```text
DL_results/
  resnet_test_performance.xlsx
  test_roc.png
  test_confusion_matrix.png
```

`resnet_test_performance.xlsx` contains test predictions and bootstrap 95% confidence intervals.

## MRI preprocessing notes

Deep-learning performance can be dominated by preprocessing choices. The current code deliberately keeps them explicit in YAML.

- `crop_to_mask: true` crops around the ROI before resizing.
- `apply_mask: true` removes voxels outside the mask after resizing.
- `normalization: zscore_mask` computes z-score statistics within the ROI when a mask exists.
- `output_size` controls GPU memory strongly; 3D ResNet-50 at `[64,128,128]` may require a small batch size.

For multi-sequence MRI, sequences should be spatially registered to the same grid before this pipeline. A simple shape check cannot prove registration, but a shape mismatch definitely means they are not directly stackable.

## Research design recommendation

Keep the independent test set locked. Architecture depth, learning rate, augmentation, preprocessing, checkpoint choice, and decision threshold should be selected using only the train-validation cohort. Once the configuration is finalized, evaluate the independent test set and report it as the final result.

For a publication-quality comparison with the previous radiomics models, keep exactly the same case-level train-validation/test assignment and report both approaches on the identical independent test cohort.
