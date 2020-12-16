# Train models
# The following command line arguments are used depending on the respective
# model and the dataset. The following model and dataset combinations are possible:
#     -model LeNet_falcon -data MNIST
#     -model VGG19_falcon -data CIFAR10
#     -model LSTM_falcon -data MNISTseq
#     -model GRU_falcon -data MNISTseq
#     -model LSTM_NLP_falcon -data Newsgroups20
#     -model ResNet50_falcon -data Imagenet
# Additional command line arguments:
#     --bool_load_model: specify whether to load the model or randomly initialize it
#     --load_path: path to where the model is loaded from
#     --save_path: path to where the model is saved
#     --path_data_imagenet: path to imagenet dataset
#     --path_data_newsgroups20: path to newsgroups20 dataset
#     --path_glove_embeddings: path to pre-trained glove embeddings
#     --epochs: training epochs
#     --batch_size: batch size for training
#     --learning_rate: overall learning rate
#     --dropout_rate: specify only if model architecture requires it
#     --lambda_l2_loss: specify only if model architecture requires it
#     --lambda_ent_loss: lambda for predictive entropy loss
#     --lambda_advcalib_loss: lambda for adversarial calibration loss
#     --probability_train_perturbation: proportion of training steps
#         with adversarial calibration loss
#     --rnn_cell_type: specify RNN-cell type: "RNN", "LSTM" or "GRU"
#     --n_units: number of units in RNN-cell
#     --n_hidden_layers: number of layers in a RNN, LSTM or GRU model
#     --input_seq_length: is needed only for NLP tasks
#     --n_vocab_words: length of vocabulary (is needed only for NLP tasks)
#     --embedding_layer_size: is needed only for NLP tasks
#     --random_seed: optionally set random seed


import os
import sys
sys.path.append("../")
import argparse
import source.data.mnist as mnist
import source.data.cifar10 as cifar10
import source.data.newsgroups20 as newsgroups20
import source.data.imagenet as imagenet
import source.Model as model_class
import source.Perturbation_Generator as perturbation_generator


parser = argparse.ArgumentParser(description="Train models")
parser.add_argument(
    "-model", nargs="?",
    help="model name e.g. LeNet_falcon")
parser.add_argument(
    "-data", nargs="?",
    help="data name e.g. MNIST")

parser.add_argument(
    "--bool_load_model",nargs="?",
    help="enter bool_load_model e.g. True or False")
parser.add_argument(
    "--load_path",nargs="?",
    help="enter load_path for to pre trained model")
parser.add_argument(
    "--save_path",
    nargs="?",help="enter save_path e.g. results/")
parser.add_argument(
    "--path_data_imagenet",nargs="?",
    help="enter path_data_imagenet e.g. datasets/imagenet/")
parser.add_argument(
    "--path_data_newsgroups20",nargs="?",
    help="enter path_data_newsgroups20 e.g. datasets/newsgroups20/")
parser.add_argument(
    "--path_glove_embeddings",nargs="?",
    help="enter path_glove_embeddings")

parser.add_argument(
    "--epochs",type=int,nargs="?",
    help="enter epochs")
parser.add_argument(
    "--batch_size",type=int,nargs="?",
    help="enter batch_size")
parser.add_argument(
    "--learning_rate",type=float,nargs="?",
    help="enter learning_rate")
parser.add_argument(
    "--dropout_rate",type=float,nargs="?",
    help="enter dropout_rate")
parser.add_argument(
    "--lambda_l2_loss",type=float,nargs="?",
    help="enter lambda_l2_loss")
parser.add_argument(
    "--lambda_ent_loss",type=float,nargs="?",
    help="enter lambda_ent_loss")
parser.add_argument(
    "--lambda_advcalib_loss",type=float,nargs="?",
    help="enter lambda_advcalib_loss")
parser.add_argument(
    "--probability_train_perturbation",type=float,nargs="?",
    help="enter probability_train_perturbation")
parser.add_argument(
    "--rnn_cell_type",nargs="?",
    help="enter rnn_cell_type")
parser.add_argument(
    "--n_units",type=int,nargs="?",
    help="enter n_units")
parser.add_argument(
    "--n_hidden_layers",type=int,nargs="?",
    help="enter n_hidden_layers")
parser.add_argument(
    "--input_seq_length",type=int,nargs="?",
    help="enter input_seq_length")
parser.add_argument(
    "--n_vocab_words",type=int,nargs="?",
    help="enter n_vocab_words")
parser.add_argument(
    "--embedding_layer_size",type=int,nargs="?",
    help="enter embedding_layer_size")
parser.add_argument(
    "--random_seed",type=int,nargs="?",
    help="enter random_seed")

args = parser.parse_args()

if args.model == None or args.data == None:
    print('Please specify -model and -data!')
    sys.exit()

def ifValueNotNone(default, value):
    """check whether value is not None (set to default if None)"""
    if value is not None:
        return value
    else:
        return default

def ifValue(value,name_value):
    """check whether value is not None (throw error if None)"""
    if value is None:
        raise ValueError(str(name_value)+' needs to be specified!')
    else:
        return value

def str2bool(value):
    """convert string to bool if possible"""
    if value in ["True", "true", "1", 1]:
        return True
    elif value in ["False", "false", "0", 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

save_path = ifValueNotNone("../results/", args.save_path)
bool_load_model = str2bool(args.bool_load_model)
if bool_load_model == True:
    load_path = ifValue(args.load_path,"load_path")
else:
    load_path = None
if args.data == 'Imagenet': path_data_imagenet = ifValue(
    args.path_data_imagenet,'path_data_imagenet')
elif args.data == 'Newsgroups20': path_data_newsgroups20 = ifValue(
    args.path_data_newsgroups20,'path_data_newsgroups20')

dict_models = {'LeNet_falcon': {'MNIST': {}},
               'VGG19_falcon': {'CIFAR10': {}},
               'LSTM_falcon': {'MNISTseq': {}},
               'GRU_falcon': {'MNISTseq': {}},
               'LSTM_NLP_falcon': {'Newsgroups20': {}},
               'ResNet50_falcon': {'Imagenet': {}}}

if args.model == 'LeNet_falcon' and args.data == 'MNIST':
    #LeNet_falcon - MNIST
    dict_models['LeNet_falcon']['MNIST']['architecture'] = 'LeNet_falcon'
    dict_models['LeNet_falcon']['MNIST']['model_name'] = 'LeNet_falcon'
    dict_models['LeNet_falcon']['MNIST']['data_name'] = 'MNIST'
    dict_models['LeNet_falcon']['MNIST']['learning_rate'] = ifValue(
        args.learning_rate,
        'learning_rate'
    )
    dict_models['LeNet_falcon']['MNIST']['lambda_ent_loss'] = ifValue(
        args.lambda_ent_loss,'lambda_ent_loss')
    dict_models['LeNet_falcon']['MNIST']['lambda_advcalib_loss'] = ifValue(
        args.lambda_advcalib_loss,'lambda_advcalib_loss')
    dict_models['LeNet_falcon']['MNIST']['epochs'] = ifValue(
        args.epochs,'epochs')
    dict_models['LeNet_falcon']['MNIST']['batch_size'] = ifValue(
        args.batch_size,'batch_size')
    dict_models['LeNet_falcon']['MNIST']['dropout_rate'] = ifValue(
        args.dropout_rate,'dropout_rate')
    dict_models['LeNet_falcon']['MNIST']['lambda_l2_loss'] = ifValue(
        args.lambda_l2_loss,'lambda_l2_loss')
    dict_models['LeNet_falcon']['MNIST']['random_seed'] = ifValue(
        args.random_seed,'random_seed')
    dict_models['LeNet_falcon']['MNIST']['probability_train_perturbation'] = ifValue(
        args.probability_train_perturbation,'probability_train_perturbation')

elif args.model == 'VGG19_falcon' and args.data == 'CIFAR10':
    #VGG19_falcon - CIFAR10
    dict_models['VGG19_falcon']['CIFAR10']['architecture'] = 'VGG19_falcon'
    dict_models['VGG19_falcon']['CIFAR10']['model_name'] = 'VGG19_falcon'
    dict_models['VGG19_falcon']['CIFAR10']['data_name'] = 'CIFAR10'
    dict_models['VGG19_falcon']['CIFAR10']['learning_rate'] = ifValue(
        args.learning_rate, 'learning_rate')
    dict_models['VGG19_falcon']['CIFAR10']['lambda_ent_loss'] = ifValue(
        args.lambda_ent_loss, 'lambda_ent_loss')
    dict_models['VGG19_falcon']['CIFAR10']['lambda_advcalib_loss'] = ifValue(
        args.lambda_advcalib_loss, 'lambda_advcalib_loss')
    dict_models['VGG19_falcon']['CIFAR10']['epochs'] = ifValue(
        args.epochs, 'epochs')
    dict_models['VGG19_falcon']['CIFAR10']['batch_size'] = ifValue(
        args.batch_size, 'batch_size')
    dict_models['VGG19_falcon']['CIFAR10']['dropout_rate'] = ifValue(
        args.dropout_rate, 'dropout_rate')
    dict_models['VGG19_falcon']['CIFAR10']['lambda_l2_loss'] = ifValue(
        args.lambda_l2_loss, 'lambda_l2_loss')
    dict_models['VGG19_falcon']['CIFAR10']['random_seed'] = ifValue(
        args.random_seed, 'random_seed')
    dict_models['VGG19_falcon']['CIFAR10']['probability_train_perturbation'] = ifValue(
        args.probability_train_perturbation, 'probability_train_perturbation')

elif args.model == 'LSTM_falcon' and args.data == 'MNISTseq':
    #LSTM_falcon - MNISTseq
    dict_models['LSTM_falcon']['MNISTseq']['architecture'] = 'RNN_falcon'
    dict_models['LSTM_falcon']['MNISTseq']['model_name'] = 'LSTM_falcon'
    dict_models['LSTM_falcon']['MNISTseq']['data_name'] = 'MNISTseq'
    dict_models['LSTM_falcon']['MNISTseq']['rnn_cell_type'] = ifValue(
        'LSTM',args.rnn_cell_type)
    dict_models['LSTM_falcon']['MNISTseq']['learning_rate'] = ifValue(
        args.learning_rate, 'learning_rate')
    dict_models['LSTM_falcon']['MNISTseq']['lambda_ent_loss'] = ifValue(
        args.lambda_ent_loss, 'lambda_ent_loss')
    dict_models['LSTM_falcon']['MNISTseq']['lambda_advcalib_loss'] = ifValue(
        args.lambda_advcalib_loss, 'lambda_advcalib_loss')
    dict_models['LSTM_falcon']['MNISTseq']['n_units'] = ifValue(
        args.n_units, 'n_units')
    dict_models['LSTM_falcon']['MNISTseq']['n_hidden_layers'] = ifValue(
        args.n_hidden_layers, 'n_hidden_layers')
    dict_models['LSTM_falcon']['MNISTseq']['epochs'] = ifValue(
        args.epochs, 'epochs')
    dict_models['LSTM_falcon']['MNISTseq']['batch_size'] = ifValue(
        args.batch_size, 'batch_size')
    dict_models['LSTM_falcon']['MNISTseq']['dropout_rate'] = ifValue(
        args.dropout_rate, 'dropout_rate')
    dict_models['LSTM_falcon']['MNISTseq']['lambda_l2_loss'] = ifValue(
        args.lambda_l2_loss, 'lambda_l2_loss')
    dict_models['LSTM_falcon']['MNISTseq']['random_seed'] = ifValue(
        args.random_seed, 'random_seed')
    dict_models['LSTM_falcon']['MNISTseq']['probability_train_perturbation'] = ifValue(
        args.probability_train_perturbation, 'probability_train_perturbation')

elif args.model == 'GRU_falcon' and args.data == 'MNISTseq':
    #GRU_falcon - MNISTseq
    dict_models['GRU_falcon']['MNISTseq']['architecture'] = 'RNN_falcon'
    dict_models['GRU_falcon']['MNISTseq']['model_name'] = 'GRU_falcon'
    dict_models['GRU_falcon']['MNISTseq']['data_name'] = 'MNISTseq'
    dict_models['GRU_falcon']['MNISTseq']['rnn_cell_type'] = ifValue(
        'GRU',args.rnn_cell_type)
    dict_models['GRU_falcon']['MNISTseq']['learning_rate'] = ifValue(
        args.learning_rate, 'learning_rate')
    dict_models['GRU_falcon']['MNISTseq']['lambda_ent_loss'] = ifValue(
        args.lambda_ent_loss, 'lambda_ent_loss')
    dict_models['GRU_falcon']['MNISTseq']['lambda_advcalib_loss'] = ifValue(
        args.lambda_advcalib_loss, 'lambda_advcalib_loss')
    dict_models['GRU_falcon']['MNISTseq']['n_units'] = ifValue(
        args.n_units, 'n_units')
    dict_models['GRU_falcon']['MNISTseq']['n_hidden_layers'] = ifValue(
        args.n_hidden_layers, 'n_hidden_layers')
    dict_models['GRU_falcon']['MNISTseq']['epochs'] = ifValue(
        args.epochs, 'epochs')
    dict_models['GRU_falcon']['MNISTseq']['batch_size'] = ifValue(
        args.batch_size, 'batch_size')
    dict_models['GRU_falcon']['MNISTseq']['dropout_rate'] = ifValue(
        args.dropout_rate, 'dropout_rate')
    dict_models['GRU_falcon']['MNISTseq']['lambda_l2_loss'] = ifValue(
        args.lambda_l2_loss, 'lambda_l2_loss')
    dict_models['GRU_falcon']['MNISTseq']['random_seed'] = ifValue(
        args.random_seed, 'random_seed')
    dict_models['GRU_falcon']['MNISTseq']['probability_train_perturbation'] = ifValue(
        args.probability_train_perturbation, 'probability_train_perturbation')

elif args.model == 'LSTM_NLP_falcon' and args.data == 'Newsgroups20':
    #LSTM_NLP_falcon - Newsgroups20
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['architecture'] = 'RNN_NLP_falcon'
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['model_name'] = 'LSTM_NLP_falcon'
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['data_name'] = 'Newsgroups20'
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['rnn_cell_type'] = ifValue(
        'LSTM',args.rnn_cell_type)
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['learning_rate'] = ifValue(
        args.learning_rate, 'learning_rate')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['lambda_ent_loss'] = ifValue(
        args.lambda_ent_loss, 'lambda_ent_loss')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['lambda_advcalib_loss'] = ifValue(
        args.lambda_advcalib_loss, 'lambda_advcalib_loss')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['n_units'] = ifValue(
        args.n_units, 'n_units')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['n_hidden_layers'] = ifValue(
        args.n_hidden_layers, 'n_hidden_layers')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['input_seq_length'] = ifValue(
        args.input_seq_length, 'input_seq_length')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['epochs'] = ifValue(
        args.epochs, 'epochs')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['batch_size'] = ifValue(
        args.batch_size, 'batch_size')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['dropout_rate'] = ifValue(
        args.dropout_rate, 'dropout_rate')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['lambda_l2_loss'] = ifValue(
        args.lambda_l2_loss, 'lambda_l2_loss')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['n_vocab_words'] = ifValue(
        args.n_vocab_words, 'n_vocab_words')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['embedding_layer_size'] = ifValue(
        args.embedding_layer_size, 'embedding_layer_size')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['path_glove_embeddings'] = ifValue(
        args.path_glove_embeddings, 'path_glove_embeddings')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['probability_train_perturbation'] = ifValue(
        args.probability_train_perturbation, 'probability_train_perturbation')
    dict_models['LSTM_NLP_falcon']['Newsgroups20']['random_seed'] = ifValue(
        args.random_seed, 'random_seed')

elif args.model == 'ResNet50_falcon' and args.data == 'Imagenet':
    #ResNet50_falcon - Imagenet
    dict_models['ResNet50_falcon']['Imagenet']['architecture'] = 'ResNet50_falcon'
    dict_models['ResNet50_falcon']['Imagenet']['model_name'] = 'ResNet50_falcon'
    dict_models['ResNet50_falcon']['Imagenet']['data_name'] = 'Imagenet'
    dict_models['ResNet50_falcon']['Imagenet']['learning_rate'] = ifValue(
        args.learning_rate, 'learning_rate')
    dict_models['ResNet50_falcon']['Imagenet']['lambda_ent_loss'] = ifValue(
        args.lambda_ent_loss, 'lambda_ent_loss')
    dict_models['ResNet50_falcon']['Imagenet']['lambda_advcalib_loss'] = ifValue(
        args.lambda_advcalib_loss, 'lambda_advcalib_loss')
    dict_models['ResNet50_falcon']['Imagenet']['epochs'] = ifValue(
        args.epochs, 'epochs')
    dict_models['ResNet50_falcon']['Imagenet']['batch_size'] = ifValue(
        args.batch_size, 'batch_size')
    dict_models['ResNet50_falcon']['Imagenet']['dropout_rate'] = ifValue(
        args.dropout_rate, 'dropout_rate')
    dict_models['ResNet50_falcon']['Imagenet']['lambda_l2_loss'] = ifValue(
        args.lambda_l2_loss, 'lambda_l2_loss')
    dict_models['ResNet50_falcon']['Imagenet']['probability_train_perturbation'] = ifValue(
        args.probability_train_perturbation, 'probability_train_perturbation')
    dict_models['ResNet50_falcon']['Imagenet']['random_seed'] = ifValue(
        args.random_seed, 'random_seed')

def get_data(data_name, params):
    """get train and valid dataset"""
    if data_name == 'MNIST':
        dataset = mnist.MNIST(train_batch_size=params['batch_size'])
        train_ds = dataset.train_ds
        valid_ds = dataset.valid_ds
    if data_name == 'MNISTseq':
        dataset = mnist.MNIST(train_batch_size=params['batch_size'],
                              flatten=True)
        train_ds = dataset.train_ds
        valid_ds = dataset.valid_ds
    if data_name == 'CIFAR10':
        dataset = cifar10.CIFAR10(train_batch_size=params['batch_size'])
        train_ds = dataset.train_ds
        valid_ds = dataset.valid_ds
    if data_name == 'Newsgroups20':
        dataset = newsgroups20.Newsgroups20(train_batch_size=params['batch_size'],
                                            max_seq_length=params['input_seq_length'],
                                            length_vocab=params['n_vocab_words'],
                                            data_path=path_data_newsgroups20,
                                            path_glove_embeddings=params['path_glove_embeddings'])
        train_ds = dataset.train_ds
        valid_ds = dataset.valid_ds
    if data_name == 'Imagenet':
        dataset = imagenet.Imagenet(train_batch_size=params['batch_size'],
                                    data_path=path_data_imagenet)
        train_ds = dataset.train_ds
        valid_ds = dataset.valid_ds
    return train_ds, valid_ds, dataset

def get_perturbation_generator(dataset, params):
    """create perturbation generator"""
    list_model_names = ['LeNet_falcon',
                        'VGG19_falcon',
                        'LSTM_falcon',
                        'GRU_falcon',
                        'LSTM_NLP_falcon',
                        'ResNet50_falcon']
    if params['model_name'] in list_model_names:
        perturb_generator = perturbation_generator.PerturbationGenerator(dataset=dataset)
    else:
        perturb_generator = None
    return perturb_generator

def train_model(params, save_path, bool_load_model=False, load_path=None):
    """general function for training a model"""
    print('Train %s...' % (params['architecture']+'_'+params['data_name']))
    train_ds, valid_ds, dataset = get_data(params['data_name'], params)
    perturb_generator = get_perturbation_generator(dataset, params)
    model = model_class.Model(
        params['architecture'],
        save_path_general=save_path+params['data_name']+'/'+params['model_name']+'/'
    )
    if bool_load_model == False:
        model.build(dataset=dataset, **params)
    else:
        model.load(load_path=load_path, dataset=dataset, **params)
    model.train(train_ds, valid_ds, perturb_generator, **params)
    model.save()
    del (model)
    print('Training of %s finished!' % (params['architecture']+'_'+params['data_name']))

#allow GPU memory growth
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
print('number physical_devices: ', len(physical_devices))

# Train Model
train_model(dict_models[args.model][args.data], save_path, bool_load_model, load_path)
