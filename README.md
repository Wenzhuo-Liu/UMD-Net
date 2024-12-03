
# UMD-Net: Unified Assistive Driving Multi-task Perception Network based on Multimodal Fusion

This repository contains the implementation of the UMD-Net, a unified assistive driving perception network designed for multi-task perception using multimodal data fusion. It includes scripts for data preprocessing, training, validation, and testing.

## Repository Structure

- `Main.py`: The main script for training, validation, and testing the UMD-Net model. Modify the `mode` parameter to switch between `train` (training and validation) and `test` modes.
- `Crop.ipynb`: A Jupyter Notebook for preprocessing the dataset. This script performs data segmentation and prepares the dataset for training and evaluation.
- `AFF_fusion.py`: Implementation of the Attention-based Feature Fusion (AFF) module used in the network.
- `GLI_CAM.py`: Implementation of the Guided Localization and Interpretation (GLI) module.
- `MS_FRF.py`: Script for Multi-scale Feature Representation Fusion.
- `attention/`: Directory containing attention-related modules.
- `requirements.txt`: List of Python dependencies required to run the project.
- `training.csv`: CSV file containing the training dataset annotations and metadata.
- `validation.csv`: CSV file containing the validation dataset annotations and metadata.
- `testing.csv`: CSV file containing the testing dataset annotations and metadata.

## Requirements

Install the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess the Dataset

Use the `Crop.ipynb` notebook to preprocess the dataset. This step involves splitting and preparing the data for training and testing.

### 2. Train and Validate the Model

To train and validate the model, run the `Main.py` script in `train` mode:

```bash
python Main.py --mode train
```

This will train the model and validate its performance on the validation dataset.

### 3. Test the Model

To test the model, switch to `test` mode:

```bash
python Main.py --mode test
```

This will evaluate the model's performance on the testing dataset.

## Dataset

The project uses the **AIDE Dataset** for training, validation, and testing. For more information and to access the dataset, search for "AIDE Dataset" online or visit its official repository.

## Contribution

Wenzhuo Liu and Yicheng Qiao contributed equally to this work.
## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries, please contact us.


