## Setup
This code requries python ≥ 3.9, pytorch ≥ 1.12.0, and pyg. Please refer to [PyTorch installation](https://pytorch.org/) and [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 
Install other required packages: `pip install -r requirements.txt`

## Datasets
### TUSZ
The TUSZ dataset is publicly available and can be accessed from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml.
We use TUSZ v1.5.2 in this study.
#### Data preprocessing
First, we resample all EEG signals in TUSZ to 200 Hz. Run: `python data/preprocess/resample_tuh.py --raw_edf_dir {dir-to-tusz-edf-files} --save_dir {dir-to-resampled-signals}`

## Model Training
The `scripts` directory contains training example. If GPU memory is insufficient, reduce the batch size and set `accumulate_grad_batches` to a value greater than 1.
### Model training on TUSZ dataset
To train TS-S4GNN on the TUSZ dataset, specify `<dir-to-resampled-signals>`, `<preproc-save-dir>`, and `<your-save-dir>` in `scripts/run_tuh.sh`, then run:`bash scripts/run_tuh.sh`.
In `train.py`, set num_classes to 1 for detection tasks, and set num_classes to 1 for classification tasks.


Note that the first time when you run this script, it will first preprocess the resampled signals by sliding a 60-s window without overlaps and save the 60-s EEG clips and seizure/non-seizure labels in PyG data object in `<preproc-save-dir>`.


