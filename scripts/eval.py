# Evaluate models and save metrics
# The following command line arguments are used depending on the respective
# model and the dataset. The following model and dataset combinations are possible:
#     -model LeNet_falcon -data MNIST
#     -model LeNet_falcon -data fashionMNIST
#     -model VGG19_falcon -data CIFAR10
#     -model LSTM_falcon -data MNISTseq
#     -model GRU_falcon -data MNISTseq
#     -model LSTM_NLP_falcon -data Newsgroups20
#     -model ResNet50_falcon -data Imagenet
#     -model ResNet50_falcon -data ObjectNet_not_imagenet
#     -model ResNet50_falcon -data ObjectNet_only_imagenet
# Additional command line arguments:
#     --load_path: path to where the model is loaded from
#     --save_general_path: path to where the model is saved
#     --path_data_imagenet: path to imagenet dataset
#     --path_data_imagenet_corrupted: path to imagenet corrupted dataset
#     --path_data_objectnet: path to objectnet dataset
#     --path_data_newsgroups20: path to newsgroups20 dataset
#     --path_glove_embeddings: path to pre-trained glove embeddings
#     --batch_size: batch size for evaluation
#     --perturbations_general_list: specify perturbations to apply to the dataset
#     --perturbations_nlp_list: specify perturbations to apply to the nlp dataset
#     --eval_metrics_list: specify evaluation metrics
#     --input_seq_length: is needed only for NLP tasks
#     --n_vocab_words: length of vocabulary (is needed only for NLP tasks)


import os
import sys
sys.path.append("../")
import argparse
import source.data.mnist as mnist
import source.data.fashionmnist as fashionmnist
import source.data.cifar10 as cifar10
import source.data.newsgroups20 as newsgroups20
import source.data.imagenet as imagenet
import source.data.objectnet as objectnet
import source.Model as model_class
import source.Perturbation_Generator as perturbation_generator
from source.Evaluator import Evaluator
from source.utilsevaluation.measures import \
    accuracy, \
    ECE, \
    neg_log_likelihood, \
    mean_entropy, \
    brier_score, \
    confidence_scores, \
    matches


parser = argparse.ArgumentParser(description="Evaluate models")
parser.add_argument(
    "-model", nargs="?",
    help="model name e.g. LeNet_basic")
parser.add_argument(
    "-data", nargs="?",
    help="data name e.g. MNIST")

parser.add_argument(
    "--load_model_path", nargs="?",
    help="enter load_model_path e.g. results/evaluation/")
parser.add_argument(
    "--save_general_path", nargs="?",
    help="enter save_general_path e.g. results/")
parser.add_argument(
    "--path_data_imagenet", nargs="?",
    help="enter path_data_imagenet e.g. datasets/imagenet/")
parser.add_argument(
    "--path_data_imagenet_corrupted", nargs="?",
    help="enter path_data_imagenet_corrupted e.g. datasets/imagenet_corrupted/")
parser.add_argument(
    "--path_data_objectnet", nargs="?",
    help="enter path_data_objectnet e.g. datasets/imagenet/")
parser.add_argument(
    "--path_data_newsgroups20", nargs="?",
    help="enter path_data_newsgroups20 e.g. datasets/newsgroups20/")
parser.add_argument(
    "--path_glove_embeddings",nargs="?",
    help="enter path_glove_embeddings")
parser.add_argument(
    "--batch_size", type=int, nargs="?",
    help="enter batch_size")
parser.add_argument(
    "--perturbations_general_list", nargs="?",
    help="enter perturbations_general_list e.g. ['None','rot_left','rot_right']")
parser.add_argument(
    "--perturbations_nlp_list", nargs="?",
    help="enter perturbations_nlp_list e.g. ['None','char_swap']")
parser.add_argument(
    "--eval_metrics_list", nargs="?",
    help="enter eval_metrics_list e.g. [ECE,brier_score]")
parser.add_argument(
    "--input_seq_length",type=int,nargs="?",
    help="enter input_seq_length")
parser.add_argument(
    "--n_vocab_words",type=int,nargs="?",
    help="enter n_vocab_words")

args = parser.parse_args()

if args.model == None or args.data == None:
    print('Please specify -model and -data!')
    sys.exit()

def ifValueNotNone(default, value):
    #Helper function
    if value is not None:
        return value
    else:
        return default

def ifValue(value,name_value):
    if value is None:
        raise ValueError(str(name_value)+' needs to be specified!')
    else:
        return value

save_general_path = ifValue(args.save_general_path, 'save_general_path')
if args.data == 'Newsgroups20': path_data_newsgroups20 = ifValue(
    args.path_data_newsgroups20, 'path_data_newsgroups20')
if args.data in ['Imagenet', "ObjectNet_only_imagenet", "ObjectNet_not_imagenet"]:
    path_data_imagenet = ifValue(
        args.path_data_imagenet, 'path_data_imagenet')
if args.data == 'Imagenet': path_data_imagenet_corrupted = ifValue(
    args.path_data_imagenet_corrupted, 'path_data_imagenet_corrupted')
if args.data == 'ObjectNet_not_imagenet': path_data_objectnet = ifValue(
    args.path_data_objectnet, 'path_data_objectnet')
if args.data == 'ObjectNet_only_imagenet': path_data_objectnet = ifValue(
    args.path_data_objectnet, 'path_data_objectnet')
if args.model == "LSTM_NLP_falcon":
    path_glove_embeddings = ifValue(args.path_glove_embeddings, 'path_glove_embeddings')
else:
    path_glove_embeddings = "./"

# evaluation takes place on these perturbations
perturbations_general = [
    "rot_left",
    "rot_right",
     "xshift",
     "yshift",
     "xyshift",
     "shear",
     "xzoom",
     "yzoom",
     "xyzoom"
]
perturbations_nlp = [
    "char_swap"
]
perturbations_imagenet_corruptions = [
    "imagenet2012_corrupted/gaussian_noise",
    "imagenet2012_corrupted/shot_noise",
    "imagenet2012_corrupted/impulse_noise",
    "imagenet2012_corrupted/defocus_blur",
    "imagenet2012_corrupted/glass_blur",
    "imagenet2012_corrupted/motion_blur",
    "imagenet2012_corrupted/zoom_blur",
    "imagenet2012_corrupted/snow",
    "imagenet2012_corrupted/frost",
    "imagenet2012_corrupted/fog",
    "imagenet2012_corrupted/brightness",
    "imagenet2012_corrupted/contrast",
    "imagenet2012_corrupted/elastic_transform",
    "imagenet2012_corrupted/pixelate",
    "imagenet2012_corrupted/jpeg_compression",
    "imagenet2012_corrupted/gaussian_blur",
    "imagenet2012_corrupted/saturate",
    "imagenet2012_corrupted/spatter",
    "imagenet2012_corrupted/speckle_noise"
]

eval_metrics = [
    accuracy,
    ECE,
    neg_log_likelihood,
    mean_entropy,
    brier_score,
    confidence_scores,
    matches
]

dict_models = {
               'LeNet_falcon': {'MNIST': {}, 'fashionMNIST': {}},
               'VGG19_falcon': {'CIFAR10': {}},
               'LSTM_falcon': {'MNISTseq': {}},
               'GRU_falcon': {'MNISTseq': {}},
               'LSTM_NLP_falcon': {'Newsgroups20': {}},
               'ResNet50_falcon': {'Imagenet': {},'ObjectNet_not_imagenet': {},
                                   'ObjectNet_only_imagenet': {}},
               }

if args.model == 'LeNet_falcon' and args.data == 'MNIST':
    # LeNet_falcon - MNIST
    dict_models['LeNet_falcon']['MNIST']['architecture'] = 'LeNet_falcon'
    dict_models['LeNet_falcon']['MNIST']['model_name'] = 'LeNet_falcon'
    dict_models['LeNet_falcon']['MNIST']['data_name'] = 'MNIST'
    dict_models['LeNet_falcon']['MNIST']['batch_size'] = ifValue(
        args.batch_size,"batch_size")
    dict_models['LeNet_falcon']['MNIST']['load_model_path'] = ifValue(
        args.load_model_path, 'load_model_path')
    dict_models['LeNet_falcon']['MNIST']['perturbations_general_list'] = ifValueNotNone(
        perturbations_general,
        [float(i) for i in b.split(",")] if args.perturbations_general_list else None)
    dict_models['LeNet_falcon']['MNIST']['eval_metrics_list'] = ifValueNotNone(
        eval_metrics,[float(i) for i in b.split(",")] if args.eval_metrics_list else None)

elif args.model == 'LeNet_falcon' and args.data == 'fashionMNIST':
    # LeNet_falcon - FMNIST
    dict_models['LeNet_falcon']['fashionMNIST']['architecture'] = 'LeNet_falcon'
    dict_models['LeNet_falcon']['fashionMNIST']['model_name'] = 'LeNet_falcon'
    dict_models['LeNet_falcon']['fashionMNIST']['data_name'] = 'fashionMNIST'
    dict_models['LeNet_falcon']['fashionMNIST']['batch_size'] = ifValue(
          args.batch_size,"batch_size")
    dict_models['LeNet_falcon']['fashionMNIST']['load_model_path'] = ifValue(
        args.load_model_path, 'load_model_path')
    dict_models['LeNet_falcon']['fashionMNIST']['perturbations_general_list'] = ifValueNotNone(
        ["None"],[float(i) for i in b.split(",")] if args.perturbations_general_list else None)
    dict_models['LeNet_falcon']['fashionMNIST']['eval_metrics_list'] = ifValueNotNone(
        eval_metrics,[float(i) for i in b.split(",")] if args.eval_metrics_list else None)

elif args.model == 'VGG19_falcon' and args.data == 'CIFAR10':
    # VGG19_falcon - CIFAR
    dict_models['VGG19_falcon']['CIFAR10']['architecture'] = 'VGG19_falcon'
    dict_models['VGG19_falcon']['CIFAR10']['model_name'] = 'VGG19_falcon'
    dict_models['VGG19_falcon']['CIFAR10']['data_name'] = 'CIFAR10'
    dict_models['VGG19_falcon']['CIFAR10']['batch_size'] = ifValue(
          args.batch_size,"batch_size")
    dict_models['VGG19_falcon']['CIFAR10']['load_model_path'] = ifValue(
        args.load_model_path, 'load_model_path')
    dict_models['VGG19_falcon']['CIFAR10']['perturbations_general_list'] = ifValueNotNone(
        perturbations_general,
        [float(i) for i in b.split(",")] if args.perturbations_general_list else None)
    dict_models['VGG19_falcon']['CIFAR10']['eval_metrics_list'] = ifValueNotNone(
        eval_metrics,[float(i) for i in b.split(",")] if args.eval_metrics_list else None)

elif args.model == 'LeNet_falcon' and args.data == 'MNISTseq':
    # LSTM_falcon - MNISTseq
    dict_models['LSTM_falcon']['MNISTseq']['architecture'] = 'RNN_falcon'
    dict_models['LSTM_falcon']['MNISTseq']['model_name'] = 'LSTM_falcon'
    dict_models['LSTM_falcon']['MNISTseq']['data_name'] = 'MNISTseq'
    dict_models['LSTM_falcon']['MNISTseq']['batch_size'] = ifValue(
          args.batch_size,"batch_size")
    dict_models['LSTM_falcon']['MNISTseq']['load_model_path'] = ifValue(
        args.load_model_path, 'load_model_path')
    dict_models['LSTM_falcon']['MNISTseq']['perturbations_general_list'] = ifValueNotNone(
        perturbations_general,
        [float(i) for i in b.split(",")] if args.perturbations_general_list else None)
    dict_models['LSTM_falcon']['MNISTseq']['eval_metrics_list'] = ifValueNotNone(
        eval_metrics,[float(i) for i in b.split(",")] if args.eval_metrics_list else None)

elif args.model == 'GRU_falcon' and args.data == 'MNISTseq':
    # GRU_falcon - MNISTseq
    dict_models['GRU_falcon']['MNISTseq']['architecture'] = 'RNN_falcon'
    dict_models['GRU_falcon']['MNISTseq']['model_name'] = 'GRU_falcon'
    dict_models['GRU_falcon']['MNISTseq']['data_name'] = 'MNISTseq'
    dict_models['GRU_falcon']['MNISTseq']['batch_size'] = ifValue(
          args.batch_size,"batch_size")
    dict_models['GRU_falcon']['MNISTseq']['load_model_path'] = ifValue(
        args.load_model_path, 'load_model_path')
    dict_models['GRU_falcon']['MNISTseq']['perturbations_general_list'] = ifValueNotNone(
        perturbations_general,
        [float(i) for i in b.split(",")] if args.perturbations_general_list else None)
    dict_models['GRU_falcon']['MNISTseq']['eval_metrics_list'] = ifValueNotNone(
        eval_metrics,[float(i) for i in b.split(",")] if args.eval_metrics_list else None)

elif args.model == 'LSTM_NLP_falcon' and args.data == 'Newsgroups20':
    # LSTM_NLP_falcon - Newsgroups20
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['architecture'] = 'RNN_NLP_falcon'
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['model_name'] = 'LSTM_NLP_falcon'
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['data_name'] = 'Newsgroups20'
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['batch_size'] = ifValue(
          args.batch_size, "batch_size")
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['input_seq_length'] = ifValue(
          args.input_seq_length, "input_seq_length")
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['n_vocab_words'] = ifValue(
          args.n_vocab_words, "n_vocab_words")
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['load_model_path'] = ifValue(
        args.load_model_path, 'load_model_path')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['perturbations_general_list'] = ifValueNotNone(
        perturbations_nlp,
        [float(i) for i in b.split(",")] if args.perturbations_general_list else None)
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['eval_metrics_list'] = ifValueNotNone(
        eval_metrics,[float(i) for i in b.split(",")] if args.eval_metrics_list else None)
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['path_glove_embeddings'] = path_glove_embeddings

elif args.model == 'ResNet50_falcon' and args.data == 'Imagenet':
    # ResNet50_falcon - Imagenet_corrupted
    dict_models['ResNet50_falcon']['Imagenet']['architecture'] = 'ResNet50_falcon'
    dict_models['ResNet50_falcon']['Imagenet']['model_name'] = 'ResNet50_falcon'
    dict_models['ResNet50_falcon']['Imagenet']['data_name'] = 'Imagenet'
    dict_models['ResNet50_falcon']['Imagenet']['batch_size'] = ifValue(
          args.batch_size,"batch_size")
    dict_models['ResNet50_falcon']['Imagenet']['load_model_path'] = ifValue(
        args.load_model_path, 'load_model_path')
    dict_models['ResNet50_falcon']['Imagenet']['perturbations_general_list'] = ifValueNotNone(
        perturbations_imagenet_corruptions,
        [float(i) for i in b.split(",")] if args.perturbations_general_list else None)
    dict_models['ResNet50_falcon']['Imagenet']['eval_metrics_list'] = ifValueNotNone(
        eval_metrics,[float(i) for i in b.split(",")] if args.eval_metrics_list else None)

elif args.model == 'ResNet50_falcon' and args.data == 'ObjectNet_not_imagenet':
    # #ResNet50_falcon - ObjectNet_not_imagenet
    dict_models['ResNet50_falcon']['ObjectNet_not_imagenet']['architecture'] = 'ResNet50_falcon'
    dict_models['ResNet50_falcon']['ObjectNet_not_imagenet']['model_name'] = 'ResNet50_falcon'
    dict_models['ResNet50_falcon']['ObjectNet_not_imagenet']['data_name'] = 'ObjectNet_not_imagenet'
    dict_models['ResNet50_falcon']['ObjectNet_not_imagenet']['batch_size'] = ifValue(
          args.batch_size,"batch_size")
    dict_models['ResNet50_falcon']['ObjectNet_not_imagenet']['load_model_path'] = ifValue(
        args.load_model_path, 'load_model_path')
    dict_models['ResNet50_falcon']['ObjectNet_not_imagenet']['perturbations_general_list'] = ifValueNotNone(
        ["None"],[float(i) for i in b.split(",")] if args.perturbations_general_list else None)
    dict_models['ResNet50_falcon']['ObjectNet_not_imagenet']['eval_metrics_list'] = ifValueNotNone(
        eval_metrics,[float(i) for i in b.split(",")] if args.eval_metrics_list else None)

elif args.model == 'ResNet50_falcon' and args.data == 'ObjectNet_only_imagenet':
    # #ResNet50_falcon - ObjectNet_only_imagenet
    dict_models['ResNet50_falcon']['ObjectNet_only_imagenet']['architecture'] = 'ResNet50_falcon'
    dict_models['ResNet50_falcon']['ObjectNet_only_imagenet']['model_name'] = 'ResNet50_falcon'
    dict_models['ResNet50_falcon']['ObjectNet_only_imagenet']['data_name'] = 'ObjectNet_only_imagenet'
    dict_models['ResNet50_falcon']['ObjectNet_only_imagenet']['batch_size'] = ifValue(
          args.batch_size,"batch_size")
    dict_models['ResNet50_falcon']['ObjectNet_only_imagenet']['load_model_path'] = ifValue(
        args.load_model_path, 'load_model_path')
    dict_models['ResNet50_falcon']['ObjectNet_only_imagenet']['perturbations_general_list'] = ifValueNotNone(
        ["None"],[float(i) for i in b.split(",")] if args.perturbations_general_list else None)
    dict_models['ResNet50_falcon']['ObjectNet_only_imagenet']['eval_metrics_list'] = ifValueNotNone(
        eval_metrics,[float(i) for i in b.split(",")] if args.eval_metrics_list else None)


def get_data(data_name, params):
    """get test dataset"""
    if data_name == 'MNIST':
        dataset = mnist.MNIST(train_batch_size=params['batch_size'])
        test_ds = dataset.test_ds
    if data_name == 'fashionMNIST':
        dataset = fashionmnist.fashionMNIST(train_batch_size=params['batch_size'])
        test_ds = dataset.test_ds
    if data_name == 'MNISTseq':
        dataset = mnist.MNIST(train_batch_size=params['batch_size'], flatten=True)
        test_ds = dataset.test_ds
    if data_name == 'CIFAR10':
        dataset = cifar10.CIFAR10(train_batch_size=params['batch_size'])
        test_ds = dataset.test_ds
    if data_name == 'Newsgroups20':
        dataset = newsgroups20.Newsgroups20(train_batch_size=params['batch_size'],
                                            max_seq_length=params['input_seq_length'],
                                            length_vocab=params['n_vocab_words'],
                                            data_path=path_data_newsgroups20,
                                            path_glove_embeddings=params['path_glove_embeddings'])
        #test_ds = dataset.test_ds
        test_ds = dataset.test_ds_raw
    if data_name == 'Imagenet':
        dataset = imagenet.Imagenet(train_batch_size=params['batch_size'],
                                    data_path=path_data_imagenet)
        test_ds = dataset.test_ds
    if data_name == 'ObjectNet_not_imagenet':
        dataset = objectnet.Objectnet(subset='not_imagenet',
                                    train_batch_size=params['batch_size'],
                                    data_path=path_data_objectnet)
        test_ds = dataset.test_ds
        dataset = imagenet.Imagenet(train_batch_size=params['batch_size'],
                                    data_path=path_data_imagenet)
    if data_name == 'ObjectNet_only_imagenet':
        dataset = objectnet.Objectnet(subset='only_imagenet',
                                    train_batch_size=params['batch_size'],
                                    data_path=path_data_objectnet)
        test_ds = dataset.test_ds
        dataset = imagenet.Imagenet(train_batch_size=params['batch_size'],
                                    data_path=path_data_imagenet)
    return test_ds, dataset

def get_path_data_corrupted(data_name):
    """get imagenet corrupted dataset"""
    if data_name == 'Imagenet':
        return path_data_imagenet_corrupted
    else:
        return False

def get_perturbation_generator(dataset, dataset_name, params):
    """create perturbation generator"""
    perturb_generator = perturbation_generator.PerturbationGenerator(
        dataset=dataset, dataset_name=dataset_name)
    return perturb_generator

def evaluate_model(params, save_general_path, **kwargs):
    """evaluate an already trained and saved model"""

    print('Evaluate %s...' % (params['architecture']+'_'+params['data_name']))
    test_ds, dataset = get_data(params['data_name'], params)
    perturb_generator = get_perturbation_generator(
        dataset=dataset, dataset_name=params['data_name'], params=params)
    path_data_corrupted = get_path_data_corrupted(params['data_name'])

    model = model_class.Model(params['architecture'],
                                data_corrupted_path=path_data_corrupted)
    model.load(params['load_model_path'], load_logits=False, params_load=kwargs,
               dataset=dataset, data_corrupted_path=path_data_corrupted, **params)

    evaluator = Evaluator(
        model,
        perturb_generator,
        data_name = params['data_name'],
        folder_path = save_general_path
    )

    print('evaluating...')
    for perturb in params['perturbations_general_list']:
        print('perturb', perturb)
        n_eps = len(perturb_generator.possible_epsilons(perturb_type=perturb))
        print('n_eps',n_eps)
        for epsilon in range(n_eps):
            for eval_metric in params['eval_metrics_list']:
                evaluator.evaluate(test_ds,
                            eval_metric,
                            perturb,
                            epsilon,
                            from_cache=True,
                            to_cache=True,
                            save_to_file=False,
                            bins_calibration=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
                print(perturb + ", " + str(epsilon) + ", " + eval_metric.__name__)
    evaluator.save()
    print(evaluator.evaluator.storage)
    print('Evaluation of %s finished!' % (params['architecture']+'_'+params['data_name']))

#allow GPU memory growth
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Evaluate Model
evaluate_model(dict_models[args.model][args.data], save_general_path)
