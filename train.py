import pandas as pd
import logging
import argparse
import os
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, LambdaCallback, TensorBoard
from keras.optimizers import SGD
from keras.utils import np_utils
from wide_resnet import WideResNet
from utils import mk_dir, load_data
from keras.preprocessing.image import ImageDataGenerator
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser

import json
from plot_history import plot_history
import configparser
from vgg16 import *
import scipy as sc
from sklearn.preprocessing import normalize
from keras.utils import multi_gpu_model
logging.basicConfig(level=logging.DEBUG)

class Schedule:
    def __init__(self, nb_epochs):
        self.epochs = nb_epochs

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return 0.1
        elif epoch_idx < self.epochs * 0.5:
            return 0.02
        elif epoch_idx < self.epochs * 0.75:
            return 0.004
        return 0.0008


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_path", type=str, required=True,
                        help="path to config file")
    parser.add_argument("--config_number", type=str, required=True,
                        help="path to config file")
    args = parser.parse_args()                        
    return args

#Metrics and Losses
def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(K.argmax(y_true, axis=-1) - K.argmax(y_pred, axis=-1)), axis=-1)

def Wasserstein(y_true, y_pred):
    y_true_new = K.cumsum(y_true, axis=-1)
    y_pred_new = K.cumsum(y_pred, axis=-1)
    return K.mean(K.abs(K.sum(K.abs(y_true_new - y_pred_new), axis=-1)), axis=-1)

def Wass1(y_true, y_pred):
    y_true_new = K.cumsum(y_true, axis=-1)
    y_pred_new = K.cumsum(y_pred, axis=-1)
    return K.mean(K.abs(y_true_new - y_pred_new), axis=-1)

def custom_loss1(y_true, y_pred):
    return K.mean(K.abs(K.sum(K.abs(y_true - y_pred), axis=-1)), axis=-1)

def custom_loss2(y_true, y_pred):
    return K.mean(K.abs(K.sum(K.square(y_true - y_pred), axis=-1)), axis=-1)

def main():
    logging.debug("Reading Configuration...")
    args = get_args()    
    Config = configparser.ConfigParser()
    Config.read(args.config_path)
    def ConfigSectionMap(section):
        dict1 = {}
        options = Config.options(section)
        for option in options:
            try:
                dict1[option] = Config.get(section, option)
                if dict1[option] == -1:
                    DebugPrint("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1

    #Loading Fixed parameters
    input_path          = ConfigSectionMap("Fixed")['input_path']
    batch_size          = int(ConfigSectionMap("Fixed")['batch_size'])
    nb_epochs           = int(ConfigSectionMap("Fixed")['nb_epochs'])
    validation_split    = float(ConfigSectionMap("Fixed")['validation_split'])
    dim                 = int(ConfigSectionMap("Fixed")['dimension'])
    use_augmentation    = bool(ConfigSectionMap("Fixed")['use_augmentation'])
    metrics_type        = ConfigSectionMap("Fixed")['metrics']
    history             = ConfigSectionMap("Fixed")['history_save_path']
    checkpoints         = ConfigSectionMap("Fixed")['checkpoint_save_path']
    logs_dir            = ConfigSectionMap("Fixed")['log_save_path']

    #Loading parameters that vary over different configurations
    config_number       = args.config_number
    distribution        = ConfigSectionMap(config_number)['distribution']
    feature_extractor   = ConfigSectionMap(config_number)['feature_extractor']
    sigma               = float(ConfigSectionMap(config_number)['sigma'])
    optimizer_type      = ConfigSectionMap(config_number)['optimizer']
    loss_type           = ConfigSectionMap(config_number)['loss']


    logging.debug("Loading data...")
    image, _, age, _, image_size, _ = load_data(input_path)
    X_data = image

    #Alter age according to distribution type
    if distribution == "GaussBins":    
        age = age[:,np.newaxis]
        lin_y = np.linspace(0,100,dim)[:,np.newaxis]
        y_data_a = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-np.square((age-lin_y.T)/(np.sqrt(2)*sigma)))
    elif distribution == "Cls":
        y_data_a = np_utils.to_categorical(age, dim)
    print(y_data_a.shape)

    data_num = len(X_data)
    #Randomly shuffle data
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    X_data = X_data[indexes]
    y_data_a = y_data_a[indexes]
    #Split in test train set
    train_num = int(data_num * (1 - validation_split))
    X_train = X_data[:train_num]
    X_test = X_data[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]

    #Choose network
    if feature_extractor == "WideResNet":
        model = WideResNet(image_size, depth=16, k=8)()
    
    #Choose optimizer
    if optimizer_type == "sgd":
        optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)

    #Choose loss
    if loss_type == "kullback_leibler_divergence":
        loss = "kullback_leibler_divergence"
    elif loss_type == "Wasserstein":
        loss = Wasserstein  
    elif loss_type == "wass1":
        loss = Wass1  
    elif loss_type == "loss1":
        loss = custom_loss1      
    elif loss_type == "loss2":
        loss = custom_loss2     
    elif loss_type == "loss3":
        loss = "mean_squared_error"                        
    elif loss_type == "loss4":
        loss = "mean_absolute_error"                        
    elif loss_type == "categorical_crossentropy":
        loss = "categorical_crossentropy"

    #Choose metric
    if metrics_type == "mean_absolute_error":
        metric = mean_absolute_error

    #Final compilation
    model.compile(optimizer=optimizer, loss=[loss], metrics=[metric])

    logging.debug("Model summary...")
    model.count_params()
    # model.summary()

    #Callbacks
    json_log = open(os.path.join(logs_dir,"{}_{}_{:.5}_{}_{}.log".format(distribution,feature_extractor,loss_type,optimizer_type,sigma)),
                    mode='wt',
                    buffering=1)
    logging_callback = LambdaCallback(
        on_train_begin=lambda logs: json_log.write(
            json.dumps({'distribution': distribution, 'feature_extractor': feature_extractor,
                         'loss_type': loss_type, 'optimizer_type': optimizer_type, 'sigma': sigma}) + '\n'),
        on_epoch_end=lambda epoch, logs: json_log.write(
            json.dumps({'epoch': epoch, 'val_mean_absolute_error': logs['val_mean_absolute_error'], 'val_loss': logs['val_loss'], 'mean_absolute_error': logs['mean_absolute_error'], 'loss': logs['loss']}) + '\n'),
        on_train_end=lambda logs: json_log.close()
    )
    callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs)),
                 ModelCheckpoint(os.path.join(checkpoints,"weights.{}_{}_{:.5}_{}_{}.hdf5".format(distribution,feature_extractor,loss_type,optimizer_type,sigma)),
                                 monitor="val_mean_absolute_error",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto"),
                 logging_callback,
                 TensorBoard(log_dir=os.path.join(logs_dir,"{}_{}_{:.5}_{}_{}/".format(distribution,feature_extractor,loss_type,optimizer_type,sigma)),
                 histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=False, write_images=False)                 ]


    logging.debug("Running training...")
    if use_augmentation:
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=0, v_h=255))
        training_generator = MixupGenerator(X_train, y_train_a, batch_size=batch_size, alpha=0.2,
                                            datagen=datagen)()
        hist = model.fit_generator(generator=training_generator,
                                   steps_per_epoch=train_num // batch_size,
                                   validation_data=(X_test, y_test_a),
                                   epochs=nb_epochs, verbose=1,
                                   callbacks=callbacks)
    else:
        hist = model.fit(X_train, y_train_a, batch_size=batch_size, epochs=nb_epochs, verbose=1, callbacks=callbacks,
                         validation_data=(X_test, y_test_a))

    logging.debug("Saving history and graphs...")
    pd.DataFrame(hist.history).to_hdf(os.path.join(history, "history.{}_{}_{:.5}_{}_{}.hdf5".format(distribution,feature_extractor,loss_type,optimizer_type,sigma)), "history")
    # plot_history(os.path.join(history, "history.{}_{}_{:.5}_{}_{}.hdf5".format(distribution,feature_extractor,loss_type,optimizer_type,sigma)))

if __name__ == '__main__':
    main()
