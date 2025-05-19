# MVANet-Multi-Stage-Video-Attention-Network-for-SELD-with-Source-Distance-Estimation
For more detailed information, please refer to the paper titled "MVANet: Multi-Stage Video Attention Network for Sound Event Localization and Detection with Source Distance Estimation".

## Installation Guide

```python
cd MVANet-SELD
pip install -r requirements.txt
```

## Data

```python
https://pan.baidu.com/s/1i4xyp2SSq6OOJnn81aRW3A?pwd=1234
```

## Extract Audio-Visual Features

Change dataset path and feature path in `utils/cls_tools/parameters.py` script

```python
dataset_dir = '../DCASE2024_SELD_dataset/'
feat_label_dir = '../DCASE2024_SELD_dataset/seld_feat_label/'

# Extract audio features
python utils/cls_tools/batch_feature_extraction.py
```

## Package in LMDB Format

Change feature path and lmdb path in `utils/lmdb_tools/convert_lmdb.py` script

```python
npy_data_dir = '../DCASE2024_SELD_dataset/seld_feat_label/foa_dev'
npy_label_dir = '../DCASE2024_SELD_dataset/seld_feat_label/foa_dev_label'
lmdb_out_dir = '../DCASE2024_SELD_dataset/seld_feat_label/foa_dev_lmdb'

# Package lmdb
python utils/lmdb_tools/convert_lmdb.py
```

## Training

Change lmdb path, ground truth label path, feature normalization file path, and training result path in `config/MVANet.yaml`

```python
data:
  train_lmdb_dir: '...'
  test_lmdb_dir: '...'
  ref_files_dir: '...'  # fold4
  norm_file: '../DCASE2024_SELD_dataset/seld_feat_label/foa_wts'
result:
  log_output_path: '...'
  log_interval: 100
  checkpoint_output_dir: '...'
  dcase_output_dir: '...'
# Load pre-trained file
model:
  pre-train: False
  # pre-train_model: 'checkpoint_epoch146_step39858'
# Model training
bash run.sh
```

Please feel free to contact me at hyhong@mail.ustc.edu.cn. if you have any questions about the implementation or encounter any issues while using the code. I'm happy to provide additional information or assistance as needed.
