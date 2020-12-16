# Towards Trustworthy Predictions from Deep Neural Networks with Fast Adversarial Calibration

This repository contains code used to run experiments in the paper "Towards Trustworthy Predictions from Deep Neural Networks with Fast Adversarial Calibration".

[link]

It contains executable training and evaluation scripts for FALCON model described in detail in the paper.

## Datasets

ImageNet as well as ObjectNet datasets are subject to special distributional rights. Thus, they need to be downloaded from official sources. All other dataset can be downloaded with the provided code. For Newsgroups20 dataset glove embeddings file is required.

## Training

All models can be trained with the `scripts/train.py` script. In the case of imagenet a path to a pre-trained model needs to be provided, which is then fine tuned.

**The following model and dataset combinations can be trained:**\
    `-model LeNet_falcon -data MNIST`\
    `-model VGG19_falcon -data CIFAR10`\
    `-model LSTM_falcon -data MNISTseq`\
    `-model GRU_falcon -data MNISTseq`\
    `-model LSTM_NLP_falcon -data Newsgroups20`\
    `-model ResNet50_falcon -data Imagenet`

**Additional command line arguments:**\
    `--bool_load_model`: Specify whether to load the model or randomly initialize it\
    `--load_path`: Path to where the model is loaded from\
    `--save_path`: Path to where the model is saved\
    `--path_data_imagenet`: Path to imagenet dataset\
    `--path_data_newsgroups20`: Path to newsgroups20 dataset\
    `--path_glove_embeddings`: Path to pre-trained glove embeddings\
    `--epochs`: Training epochs\
    `--batch_size`: Batch size for training\
    `--learning_rate`: Overall learning rate\
    `--dropout_rate`: Specify only if model architecture requires it\
    `--lambda_l2_loss`: Specify only if model architecture requires it\
    `--lambda_ent_loss`: Lambda for predictive entropy loss\
    `--lambda_advcalib_loss`: Lambda for adversarial calibration loss\
    `--probability_train_perturbation`: Proportion of training steps with adversarial calibration loss\
    `--rnn_cell_type`: Specify RNN-cell type: "RNN", "LSTM" or "GRU"\
    `--n_units`: Number of units in RNN-cell\
    `--n_hidden_layers`: Number of layers in a RNN, LSTM or GRU model\
    `--input_seq_length`: Is needed only for NLP tasks\
    `--n_vocab_words`: Length of vocabulary (is needed only for NLP tasks)\
    `--embedding_layer_size`: Is needed only for NLP tasks\
    `--random_seed`: Optionally set random seed

## Evaluation

Evaluation of the trained models is done with the `scripts/eval.py` script. All specified metrics are stored.

**The following model and dataset combinations can be evaluated:**\
    `-model LeNet_falcon -data MNIST`\
    `-model LeNet_falcon -data fashionMNIST`\
    `-model VGG19_falcon -data CIFAR10`\
    `-model LSTM_falcon -data MNISTseq`\
    `-model GRU_falcon -data MNISTseq`\
    `-model LSTM_NLP_falcon -data Newsgroups20`\
    `-model ResNet50_falcon -data Imagenet`\
    `-model ResNet50_falcon -data ObjectNet_not_imagenet`\
    `-model ResNet50_falcon -data ObjectNet_only_imagenet`

**Additional command line arguments:**\
    `--load_path`: Path to where the model is loaded from\
    `--save_general_path`: Path to where the model is saved\
    `--path_data_imagenet`: Path to imagenet dataset\
    `--path_data_imagenet_corrupted`: Path to imagenet corrupted dataset\
    `--path_data_objectnet`: Path to objectnet dataset\
    `--path_data_newsgroups20`: Path to newsgroups20 dataset\
    `--path_glove_embeddings`: Path to pre-trained glove embeddings\
    `--batch_size`: Batch size for evaluation\
    `--perturbations_general_list`: Specify perturbations to apply to the dataset\
    `--perturbations_nlp_list`: Specify perturbations to apply to the nlp dataset\
    `--eval_metrics_list`: Specify evaluation metrics\
    `--input_seq_length`: Is needed only for NLP tasks\
    `--n_vocab_words`: Length of vocabulary (is needed only for NLP tasks)


## Folder structure

The repository is structured with the following folders.

### scripts

Training and evaluation scripts.

### source

The python source code for all the class systems implemented in this project.

`source/data`: Data classes used in the project.\
`source/models`: Implementations of FALCON models.\
`source/utilsevaluation`: All scripts that act as helpers for evaluation.\
`source/utils`: Utility functions that are used in the scripts.

`source/Evaluator.py`: Class that handels evaluation of models.\
`source/generator.py`: Computes perturbations from data.\
`source/model_factory.py`: Class that handels training and prediction.
