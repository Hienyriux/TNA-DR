# TNA-DR
Multi-Label Classification with Dual Tail-Node Augmentation for Drug Repositioning

## Requirements
* python == 3.8.10
* pytorch == 1.9.1
* scipy == 1.7.1
* scikit-learn == 1.0.2
* paddlepaddle == 2.6.0 (Optional)
* paddlenlp == 2.6.1 (Optional)

Our code should be compatible with the newer versions of these dependencies, so there is no need to worry too much about the version issues of the dependencies.

PaddlePaddle related packages are optional. We use them to create name embedding of drugs and diseases in `repoDB` dataset. If you do not plan to run this part of the code （`utils_name_emb.py`）, you don't need to install these optional packages.

## Usage

### Data Preprocessing
Please download the dataset and model files and unzip them to the root directory of this project. If you successfully downloaded and unzipped the files, you can skip the following data preprocessing steps and directly run the commands that will be introduced later to train or test the model. If you want to generate those files from scratch, you should first download the raw data files of the datasets: [Fdataset](https://github.com/BioinformaticsCSU/BNNR/blob/master/Datasets/Fdataset.mat), [Cdataset](https://github.com/BioinformaticsCSU/BNNR/blob/master/Datasets/Cdataset.mat), all files with a prefix of `lrssl` in the [directory](https://github.com/TheWall9/DRHGCN/tree/main/dataset/drimc), and download `full.csv` from [repoDB](http://apps.chiragjpgroup.org/repoDB). Then, move all the downloaded files to the `dataset/raw_data` directory of this repository, and rename the downloaded `full.csv` to `repodb_full.csv`. Then, run the code in `utils_preprocess.py`. When you run `utils_preprocess.py`, you should sequentially uncomment one code line at the end of the file (from `process_dataset("Fdataset")`) while keeping other lines commented, and then run the code once. Please note: After running the code lines related to dataset split, due to the variations of random number generators on different machines and environments, you may not get dataset splits that are exactly the same as what we provided.

If you want to recreate name embedding of drugs and diseases in `repoDB` dataset, please make sure PaddlePaddle related packages are correctly installed. Also, you should make sure `drug_id_to_name.json` and `disease_id_to_name.json` exist in `dataset/repodb` (by running the code line `process_repodb()`). You can also try using HuggingFace `Transformers`, which has similar functionality to `PaddleNLP`.

### Training, Validation, and Test
You can train the model using the following command:

For `Fdataset`:
`python tna_dr.py --dataset_name Fdataset --mode train --cv_ind 0,1,2,3,4,5,6,7,8,9 --num_folds 10 --num_epochs 1600 --pre_mp_dropout 0.8 --post_mp_dropout 0.5 --pred_dropout 0.8 --aug_type_knn disease --aug_type_contra drug_rep --tail_threshold_knn 0.0 --tail_threshold_contra 0.0 --num_neighbors 5 --lamb_knn 0.1 --lamb_contra 0.5`

For `Cdataset`:
`python tna_dr.py --dataset_name Cdataset --mode train --cv_ind 0,1,2,3,4,5,6,7,8,9 --num_folds 10 --num_epochs 1100 --pre_mp_dropout 0.8 --post_mp_dropout 0.5 --pred_dropout 0.8 --aug_type_knn disease --aug_type_contra drug_rep --tail_threshold_knn 0.0 --tail_threshold_contra 0.0 --num_neighbors 5 --lamb_knn 0.1 --lamb_contra 0.1`

For `lrssl`:
`python tna_dr.py --dataset_name lrssl --mode train --cv_ind 0,1,2,3,4,5,6,7,8,9 --num_folds 10 --num_epochs 1500 --pre_mp_dropout 0.8 --post_mp_dropout 0.0 --pred_dropout 0.8 --aug_type_knn disease --aug_type_contra drug_rep --tail_threshold_knn 0.0 --tail_threshold_contra 0.0 --num_neighbors 20 --lamb_knn 0.1 --lamb_contra 0.5`

For `repodb`:
`python tna_dr.py --dataset_name repodb --mode train --cv_ind 0,1,2,3,4,5,6,7,8,9 --num_folds 10 --num_epochs 1000 --pre_mp_dropout 0.8 --post_mp_dropout 0.5 --pred_dropout 0.8 --aug_type_knn disease --aug_type_contra drug_rep --tail_threshold_knn 0.0 --tail_threshold_contra 0.0 --num_neighbors 5 --lamb_knn 0.1 --lamb_contra 0.5`

Here's the explanation of the arguments:
* `--dataset_name`: the name of the dataset you want to use, and it can be `Fdataset`, `Cdataset`, `lrssl`, or `repodb`.
* `--mode`: running mode, which can be `train`, `test`, or `train_valid`, for training, test, or training-and-validation, respectively.
* `--cv_ind`: indices of cross validation experiments, e.g., `0` means the 0th k-fold cross validation experiment. We conduct 10 times 10-fold CV in our experiments, so `--cv_ind` can be 0-9. If you want conduct multiple CV experiments, you can concatenate the indices using ',', e.g., `0,1,2` for 0th, 1st, and 2nd CV experiments. Note that you can only specify `--cv_ind` as `0`, if you conduct experiments on new diseases (inductive settings).
* `--num_folds`: the number of cross validation folds, which can be `10` (most settings) or `5` (only for inductive settings).
* `--num_epochs`: the number of epochs you want to run for training or cross validation. This argument is ignored in `test` mode.
* `--pre_mp_dropout`, `--post_mp_dropout`, `--pred_dropout`: the dropout rate before and after the message passing process in the bipartite GNN, and the dropout rate in the prediction MLP layer.
* `--aug_type_knn`: the augmentation type for kNN tail-node augmentation, which can be `drug` or `disease`, and the default value is `disease`.
* `--aug_type_contra`: the augmentation type for contrastive tail-node augmentation, which can be `drug_rep`, `drug_pred`, or `disease`, and the default value is `drug_rep`.
* `--tail_threshold_knn`, `--tail_threshold_contra`: the tail node filtering threshold of kNN tail-node augmentation or contrastive tail-node augmentation, which can be `[0,1]`, and the default value is `0`. `0` means to only augment isolated nodes.
* `--num_neighbors`: the number of neighbors in kNN tail-node augmentation, which can be `[0, +∞]`. If the value is set to `0` or exceeds the number of nodes in the dataset, the model will treat all the (intra-domain) nodes in the dataset as neighbors.
* `--lamb_knn`, `--lamb_contra`: the loss weights of kNN tail-node augmentation or contrastive tail-node augmentation. `0` means the augmentation is deactivated.

Other optional arguments:
* `--lr`: the learning rate.
* `--seed`: the random seed.
* `--device`: the device on which the model will be run, e.g., `cuda`, `cuda:1`, `cpu`, etc. We recommend you to run the model on GPU to avoid unnecessary trouble.

The model will be saved in the directory `model`. The file names of the model weights are in the format: `{dataset_name}_{cv_idx}_{fold_idx}.pt`.

You can test the model by shifting the `--mode` argument to `test`. The code will load the model weights from `model/{dataset_name}_{cv_idx}_{fold_idx}.pt`, and then test the loaded model in the specified dataset.

The `train_valid` mode is used for hyperparameter tuning. You can set the `--mode` argument as `train_valid` to enable this mode.
