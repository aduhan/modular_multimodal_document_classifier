import argparse
import os
import pickle
from glob import glob
from typing import Callable, List, Tuple

import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from transformers import DistilBertTokenizer, TFDistilBertModel

from utils import get_paths

if not os.path.isdir('models'):
    os.mkdir("models")
    
if not os.path.isdir('validation_predictions'):
    os.mkdir("validation_predictions")
    
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_path = "data/labels/train.txt"
val_path = "data/labels/val.txt"
test_path = "data/labels/val.txt"
base_path = "data/images/"
    
num_classes = 16

train_paths = pd.read_csv(train_path, header=None)[0].apply(lambda x: x.split(" "))
val_paths = pd.read_csv(val_path, header=None)[0].apply(lambda x: x.split(" "))
test_paths = pd.read_csv(test_path, header=None)[0].apply(lambda x: x.split(" "))

# the following file from the test set was found to be corrupt
corrupt_file_path = "imagese/e/j/e/eje42e00/2500126531_2500126536.tif"
test_paths = test_paths[test_paths.apply(lambda x: corrupt_file_path not in x)].reset_index(drop=True)

train_paths = train_paths[0:128]
val_paths = val_paths[0:64]


def transform_image(file_path: str, args: argparse.Namespace) -> np.array:
    """Transfrom an image according to the model

    Args:
        file_path: path of a file
        args: the arguments input
        
    Returns:
        np.ndarray: a 3-dimensional normalized array
    """
    
    holistic = np.asarray(Image.open(file_path).resize((720, 936), Image.ANTIALIAS))/255
    top_to_body_pixels = int((holistic.shape[0]-480)/2)

    if args.model_name == "holistic":
        holistic = cv2.resize(holistic, args.image_shape[:2])
        holistic = np.repeat(holistic[:, :, np.newaxis], 3, axis=2)
        
        return holistic

    elif args.model_name == "header":
        header = holistic[:307,:]
        header = cv2.resize(header, args.image_shape[:2])
        header = np.repeat(header[:,:, np.newaxis], 3, axis=2)
        
        return header

    elif args.model_name == "footer":
        footer = holistic[holistic.shape[0]-307:,:]
        footer = cv2.resize(footer, args.image_shape[:2])
        footer = np.repeat(footer[:,:, np.newaxis], 3, axis=2)
        
        return footer
    
    elif args.model_name == "left_body":
        left_body = holistic[top_to_body_pixels:holistic.shape[0]-top_to_body_pixels, :360]
        left_body = cv2.resize(left_body, args.image_shape[:2])
        left_body = np.repeat(left_body[:,:, np.newaxis], 3, axis=2)
        
        return left_body

    elif args.model_name == "right_body":
        right_body = holistic[top_to_body_pixels:holistic.shape[0]-top_to_body_pixels, 360:]
        right_body = cv2.resize(right_body, args.image_shape[:2])
        right_body = np.repeat(right_body[:,:, np.newaxis], 3, axis=2)
        
        return right_body
    
def transform_text(X: List, args: argparse.Namespace) -> Tuple:
    """Generate the input for the Distilbert model.
    
    Args:
        X: a list of text
        args: the arguments input
        
    Returns:
        input: a list of string to id mappings
        attention_mask: a list of zeros and ones
    """
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    input = []
    attention_masks = []
    
    for x in X:
        train_dbert_inps=tokenizer.encode_plus(x, add_special_tokens=True, max_length=args.max_len, pad_to_max_length=True, return_attention_mask=True, truncation=True)
        input.append(train_dbert_inps['input_ids'])
        attention_masks.append(train_dbert_inps['attention_mask'])  
    
    input = np.array(input)
    attention_masks = np.array(attention_masks)  
        
    return input, attention_masks
    
    
def create_dataset_generator(file_paths: pd.Series, args: argparse.Namespace) -> Callable[[], Tuple]:
    """Need to make an argument-free generator as a function inside a function with arguments.
    
    Args:
        file_paths: paths of files
        model: the NN model
        
    Returns:
        dataset_gen: a generator function
    """

    def dataset_gen():
        for file_path in file_paths:
            img = transform_image(base_path + file_path[0], args)
                            
            yield (img, file_path[1])
            
    return dataset_gen


def configure_for_performance(dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    """Optimize for performance,
    by caching and prefetching.
    
    Args:
        dataset: a TF dataset
        batch_size: batch size
        
    Returns:
        dataset: an optimized TF dataset
    """

    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset
            

def build_image_model(args: argparse.Namespace) -> keras.engine.functional.Functional:
    """The base model is initialized here and then 
    a classification head is put on top.
    
    Args:
        args: the arguments input
    """
    
    inputs = layers.Input(shape=args.image_shape)

    dropout = float(args.image_dropout)
    num_dense_neurons = int(args.image_num_dense_neurons)
    learning_rate = float(args.image_classification_head_learning_rate)

    model = EfficientNetB1(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout, name="top_dropout")(x)
    x = layers.Dense(num_dense_neurons, activation='relu')(x)
    outputs = layers.Dense(16, activation="softmax", name="pred")(x) # num_classes

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    
    return model


def build_text_model(args: argparse.Namespace)-> keras.engine.functional.Functional:
    """The base model is initialized here and then 
    a classification head is put on top.
    
    Args:
        args: the arguments input
    """
    
    dropout = float(args.text_dropout)
    num_dense_neurons = int(args.text_num_dense_neurons)
    learning_rate = float(args.text_classification_head_learning_rate)
    max_len = int(args.max_len)

    dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    
    # Freeze the pretrained weights
    dbert_model.trainable = False

    inps = layers.Input(shape = (max_len,), dtype='int64')
    masks = layers.Input(shape = (max_len,), dtype='int64')
    
    dbert_layer = dbert_model.distilbert(inps, attention_mask=masks)[0][:,0,:] # extract representation for the [CLS] token
    
    dense = layers.Dense(num_dense_neurons,activation='relu')(dbert_layer)
    dropout= layers.Dropout(dropout)(dense)
    pred = layers.Dense(16, activation='softmax')(dropout)
    
    model = tf.keras.Model(inputs=[inps,masks], outputs=pred)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
       
    model.compile(optimizer=optimizer,                                     
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])

    return model


def unfreeze_model(model: keras.engine.functional.Functional,
                   learning_rate: float) -> keras.engine.functional.Functional:
    """Unfreeze all layers of a model.
    
    Args:
        model: the NN model
        learning_rate: a new learning rate after unfreezing all weights
        
    Returns:
        model: the unfrozen NN model
    """
    
    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model



def train_image(args: argparse.Namespace):
    """The image training is done here.
    Set up TF Datasets for training and validation,
    and the corresponding training regime.
    When training the holistic model, first train the head 
    and then unfreeze the whole model and continue training. 
    When training the remaining models (header, footer,
    left body, right body), load the holistic model and 
    fine tune.

    Args:
        args: the arguments input
    """
    
    train_paths, val_paths, _, y_val, _ = get_paths(args.model_name)

    es = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False
    )
    
    train_generator = create_dataset_generator(train_paths, args)
    val_generator = create_dataset_generator(val_paths, args)

    train_ds = tf.data.Dataset.from_generator(train_generator, 
                                            output_types=(tf.float32, tf.int32),
                                            output_shapes=(args.image_shape, ([])))

    val_ds = tf.data.Dataset.from_generator(val_generator, 
                                            output_types=(tf.float32, tf.int32),
                                            output_shapes=(args.image_shape, ([])))

    train_ds = configure_for_performance(train_ds, int(args.image_batch_size))
    val_ds = configure_for_performance(val_ds, int(args.image_batch_size))

    
    if args.model_name == "holistic":
        model = build_image_model(args)
        
        filepath = f"models/{args.model_name}_head.hdf5"
        cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        model.fit(train_ds, batch_size=int(args.image_batch_size), epochs=int(args.image_epochs), validation_data=val_ds, callbacks=[es, cp]) 
        model = load_model(filepath)
        
        new_learning_rate = float(args.image_learning_rate)
        model = unfreeze_model(model, new_learning_rate)
        
        os.remove(filepath)
    
    else:
        model = load_model("models/holistic.hdf5")
        
    filepath = f"models/{args.model_name}.hdf5"
    cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    model.fit(train_ds, batch_size=int(args.image_batch_size), epochs=int(args.image_epochs), validation_data=val_ds, callbacks=[es, cp])
    
    predictions = model.predict(val_ds)
    meta_classifier_data = [predictions, y_val]
    
    with open(f"validation_predictions/{args.model_name}.pkl", "wb") as f:
        pickle.dump(meta_classifier_data, f)
            

def train_text(args: argparse.Namespace):
    """The text training is done here.
    Set up TF Datasets for training and validation.

    Args:
        args: the arguments input
    """

    es = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False
    )
    
    X_train, X_val, _, y_train, y_val, _ = get_paths(args.model_name)
    
    train_input, train_attention_masks = transform_text(X_train, args)
    val_input, val_attention_masks = transform_text(X_val, args)
    
    model = build_text_model(args)
    
    filepath = f"models/{args.model_name}_head.hdf5"
    cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    model.fit([train_input,train_attention_masks],np.array(y_train), 
              batch_size=int(args.text_batch_size),
              epochs=int(args.text_epochs), 
              validation_data=([val_input,val_attention_masks],np.array(y_val)),
              callbacks=[es, cp]) 
    
    model = load_model(filepath)
    
    new_learning_rate = float(args.text_learning_rate)
    model = unfreeze_model(model, new_learning_rate)
    
    os.remove(filepath)
    
    filepath = f"models/{args.model_name}.hdf5"
    cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    model.fit([train_input,train_attention_masks],np.array(y_train), 
              batch_size=int(args.text_batch_size),
              epochs=int(args.text_epochs), 
              validation_data=([val_input,val_attention_masks],np.array(y_val)), 
              callbacks=[es, cp])  
    
    
    predictions = model.predict([val_input,val_attention_masks],batch_size=int(args.text_batch_size))
    meta_classifier_data = [predictions, y_val]
    
    with open("validation_predictions/distilbert.pkl", "wb") as f:
        pickle.dump(meta_classifier_data, f)
        
def train_meta(args: argparse.Namespace):
    """The meta classifier training is done here.
    Load the predictions from all 5 or 6 base classifiers,
    depending on the input argument, construct and train 
    a meta classifier (simple neural network).
    
    Args:
        args: the arguments input
    """
    
    es = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=5,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )

    model = Sequential()
    
    meta_classifier_data_paths = glob("validation_predictions/*")
    meta_classifier_data_paths = [path for path in meta_classifier_data_paths if path.endswith("pkl")]

    data_order = ["holistic", "header", "footer", "left_body", "right_body", "distilbert"]

    all_data = []
    
    for order in data_order:
        for path in meta_classifier_data_paths:
            if order in path:
                with open(path, "rb") as f:
                    all_data.append(pickle.load(f))
    
    if args.image_only:
        filepath = f"models/{args.model_name}_image_only.hdf5"
        num_dense_neurons = 128
        learning_rate = 0.0005
        del all_data[-1]
    else:
        filepath = f"models/{args.model_name}.hdf5"
        num_dense_neurons = 1024
        learning_rate = 0.00005
    
    dataset = all_data[0][0]
    for data in all_data[1:]:
        dataset = np.concatenate((dataset, data[0]), axis=1)
    
    # Take 90% of the validation dataset predictions as training set
    x_train = dataset[4000:]
    x_val = dataset[:4000]
    
    y_train = all_data[0][1][4000:]
    y_val = all_data[0][1][:4000]
        
    model.add(layers.Dense(num_dense_neurons, input_dim=dataset.shape[1], activation='relu'))
    model.add(layers.Dropout(0.3))
    
    for _ in range(3):
        model.add(layers.Dense(num_dense_neurons, activation='relu'))
        model.add(layers.Dropout(0.3))
        
    model.add(layers.Dense(16, activation='softmax'))
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # compile the model
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    model.fit(x_train, np.array(y_train), validation_data=(x_val, np.array(y_val)), epochs=50, batch_size=32, callbacks=[es, cp])

def main():
    """The main function.
    Usage: python train_base.py -model_name -[model_name] -epochs [epochs] -image_only
    """
    
    parser = argparse.ArgumentParser("""For an image only model, train all except 'distilbert'. Start with the 'holistic' model, 
    then subsequently train the 'header', 'footer', 'left_body' and 'right_body'. Optionally,
    train 'distilbert', if -image_only is set to False. The default hyperparameters are set such that the highest test set accuracy on RVL-CDIP is achieved.\n""")

    # Required argument
    parser.add_argument('-model_name', required=True, help="model_name is either 'holistic', 'header', 'footer', 'left_body', 'right_body' or 'distilbert'")
    
    # Optional model architecture argument
    parser.add_argument('-image_only', action=argparse.BooleanOptionalAction, default=False)

    # Optional image model hyperparameter arguments
    parser.add_argument('-image_epochs', default=200)   
    parser.add_argument('-image_classification_head_learning_rate', default=0.0001)  
    parser.add_argument('-image_learning_rate', default=0.0001)  
    parser.add_argument('-image_dropout', default=0.3)   
    parser.add_argument('-image_num_dense_neurons', default=50)
    parser.add_argument('-image_batch_size', default=32)   
    
    # Optional text model hyperparameter arguments
    parser.add_argument('-text_epochs', default=200)   
    parser.add_argument('-text_classification_head_learning_rate', default=0.0001)  
    parser.add_argument('-text_learning_rate', default=0.0001)  
    parser.add_argument('-text_dropout', default=0.3)   
    parser.add_argument('-text_num_dense_neurons', default=512)
    parser.add_argument('-text_batch_size', default=32)   
     
    # Optional preprocessing hyperparameter arguments
    parser.add_argument('-image_shape', default=(384,384,3))
    parser.add_argument('-max_len', default=256)
     
    args = parser.parse_args()    
    
    assert args.model_name == "holistic" or args.model_name == "header" or args.model_name == "footer" or args.model_name == "left_body" or args.model_name == "right_body" or args.model_name == "distilbert" or args.model_name == "meta_classifier", "model_name must be set to one of ['holistic', 'header', 'footer', 'left_body', 'right_body', 'distilbert', 'meta_classifier']"   
    
    if args.model_name == "meta_classifier":
        assert len(os.listdir("validation_predictions")) == 5 or len(os.listdir("validation_predictions")) == 6, "first train all 5/6 base models (holistic, header, footer, left_body, right_body and optionally distilbert - depending on if an image only system is trained), then train the meta_classifier"
    
    
    if args.model_name == "distilbert":
        train_text(args) 
    elif args.model_name == "meta_classifier":
        train_meta(args)
    else:
        train_image(args)
            
if __name__ == "__main__":
    main()
