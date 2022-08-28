import argparse
import json
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
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, Sequential, load_model
from transformers import DistilBertTokenizer, TFDistilBertModel

if not os.path.isdir('fine_tuned_models'):
    os.mkdir("fine_tuned_models")
    
if not os.path.isdir('fine_tuned_predictions'):
    os.mkdir("fine_tuned_predictions")
    
AUTOTUNE = tf.data.experimental.AUTOTUNE

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
            img = transform_image(file_path[0], args)
                            
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
    
    model = load_model(f"models/{args.model_name}.hdf5")
    model = Model(model.input, model.layers[-6].output)
    inputs = model.input
    model.trainable = False      

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.Dropout(args.image_dropout, name="top_dropout")(x)
    x = layers.Dense(args.image_num_dense_neurons, activation='relu')(x)
    outputs = layers.Dense(args.num_classes, activation="softmax", name="pred")(x) # num_classes

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.image_learning_rate)

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
    learning_rate = float(args.text_learning_rate)
    max_len = int(args.max_len)

    dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    
    # Freeze the pretrained weights
    dbert_model.trainable = False

    inps = layers.Input(shape = (max_len,), dtype='int64')
    masks = layers.Input(shape = (max_len,), dtype='int64')
    
    dbert_layer = dbert_model.distilbert(inps, attention_mask=masks)[0][:,0,:] # extract representation for the [CLS] token
    
    dense = layers.Dense(num_dense_neurons,activation='relu')(dbert_layer)
    dropout= layers.Dropout(dropout)(dense)
    pred = layers.Dense(args.num_classes, activation='softmax')(dropout)
    
    model = tf.keras.Model(inputs=[inps,masks], outputs=pred)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
       
    model.compile(optimizer=optimizer,                                     
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])

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
    
    # read and shuffle the data
    train_paths = pd.read_csv(args.train_labels_paths, header=None)[0].apply(lambda x: x.split(" "))
    train_paths = train_paths.sample(frac=1).reset_index(drop=True)
    y_train = list(train_paths.apply(lambda x: int(x[1])))

    validation_set = False
    
    if not isinstance(args.val_labels_paths, type(None)):
        val_paths_df = pd.read_csv(args.val_labels_paths, header=None).sample(frac=1).reset_index(drop=True)
        val_paths = val_paths_df[0].apply(lambda x: x.split(" "))
        y_val = list(val_paths_df[0].apply(lambda x: int(x.split(" ")[1])))
        
        validation_set = True
    
    # if no path for validation labels is given and the number of examples per class is greater than min_dataset_size_for_val, take 10% of the training set for validation
    elif isinstance(args.val_labels_paths, type(None)) and (pd.DataFrame.from_records([s[1] for s in train_paths]).value_counts() > int(args.min_dataset_size_for_val)).all():
        temp_df = pd.DataFrame.from_records([(row[0], row[1]) for row in train_paths])
        train_paths, val_paths = train_test_split(temp_df, test_size=0.1, stratify=temp_df[1], random_state=1) 
        
        train_paths = pd.Series(train_paths[0] + " " + train_paths[1]).apply(lambda x: x.split(" "))
        val_paths = pd.Series(val_paths[0] + " " + val_paths[1]).apply(lambda x: x.split(" "))
        
        validation_set = True
    
    train_generator = create_dataset_generator(train_paths, args)
    train_ds = tf.data.Dataset.from_generator(train_generator, 
                                            output_types=(tf.float32, tf.int32),
                                            output_shapes=(args.image_shape, ([])))
    
    train_ds = configure_for_performance(train_ds, int(args.image_batch_size))
    
    model = build_image_model(args)
    filepath = f"fine_tuned_models/{args.model_name}.hdf5"
    
    if validation_set:
        val_generator = create_dataset_generator(val_paths, args)

        val_ds = tf.data.Dataset.from_generator(val_generator, 
                                                output_types=(tf.float32, tf.int32),
                                                output_shapes=(args.image_shape, ([])))

        val_ds = configure_for_performance(val_ds, int(args.image_batch_size))
                
        es = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=3,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False
        )
    
        cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        model.fit(train_ds, batch_size=int(args.image_batch_size), epochs=int(args.image_epochs), validation_data=val_ds, callbacks=[es, cp])
        
        predictions = model.predict(val_ds)
        meta_classifier_data = [predictions, y_val]
        
        with open(f"fine_tuned_predictions/validation_{args.model_name}.pkl", "wb") as f:
            pickle.dump(meta_classifier_data, f)
    
    else:
        model.fit(train_ds, batch_size=int(args.image_batch_size), epochs=int(args.image_epochs_if_no_val_set))
        model.save(filepath)
    
        predictions = model.predict(train_ds)
        meta_classifier_data = [predictions, y_train]
        
        with open(f"fine_tuned_predictions/train_{args.model_name}.pkl", "wb") as f:
            pickle.dump(meta_classifier_data, f)
            

def train_text(args: argparse.Namespace):
    """The text training is done here.
    Set up TF Datasets for training and validation.

    Args:
        args: the arguments input
    """
    
    validation_set = False
    
    if len(os.listdir("fine_tuned_data")) == 2:
        with open("fine_tuned_data/train.json", "r") as train:
            X_train = json.load(train)
            
        with open("fine_tuned_data/train_labels.json", "r") as train:
            y_train = json.load(train)
        
        X_train = list(X_train.values())
        y_train = list(pd.Series(y_train.values()).astype(int))
        
    else:
        with open("fine_tuned_data/val.json", "r") as val:
            X_val = json.load(val)
        
        with open("fine_tuned_data/val_labels.json", "r") as val:
            y_val = json.load(val)   
            
        X_val = list(X_val.values())
        y_val = list(pd.Series(y_val.values().astype(int)))
        validation_set = True     

    train_input, train_attention_masks = transform_text(X_train, args)
    
    model = build_text_model(args)
    
    filepath = f"fine_tuned_models/{args.model_name}.hdf5"

    if validation_set:
        val_input, val_attention_masks = transform_text(X_val, args)
        
        es = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=3,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False
        )
        
        cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        model.fit([train_input,train_attention_masks],np.array(y_train), 
                batch_size=int(args.text_batch_size),
                epochs=int(args.text_epochs), 
                validation_data=([val_input,val_attention_masks],np.array(y_val)),
                callbacks=[es, cp])
        
        predictions = model.predict([val_input,val_attention_masks],batch_size=int(args.text_batch_size))
        meta_classifier_data = [predictions, y_val]
        
        with open("fine_tuned_predictions/validation_distilbert.pkl", "wb") as f:
            pickle.dump(meta_classifier_data, f)
        
    else: 
        model.fit([train_input,train_attention_masks],np.array(y_train), 
            batch_size=int(args.text_batch_size),
            epochs=int(args.text_epochs_if_no_val_set))        
        
        model.save(filepath)
    
        predictions = model.predict([train_input,train_attention_masks],batch_size=int(args.text_batch_size))
        meta_classifier_data = [predictions, y_train]
    
        with open("fine_tuned_predictions/train_distilbert.pkl", "wb") as f:
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
        patience=3,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )

    model = Sequential()
    
    meta_classifier_data_paths = glob("fine_tuned_predictions/*")
    validation_set = False

    train_paths = pd.read_csv(args.train_labels_paths, header=None)[0].apply(lambda x: x.split(" "))
    if not isinstance(args.val_labels_paths, type(None)) or (isinstance(args.val_labels_paths, type(None)) and 
                                                             (pd.DataFrame.from_records([s[1] for s in train_paths]).value_counts() > int(args.min_dataset_size_for_val)).all()):
        validation_set = True

    if validation_set:    
        meta_classifier_data_paths = [path for path in meta_classifier_data_paths if path.endswith("pkl") and "validation" in path]
    else:
        meta_classifier_data_paths = [path for path in meta_classifier_data_paths if path.endswith("pkl") and "train" in path]

    data_order = ["holistic", "header", "footer", "left_body", "right_body", "distilbert"]

    all_data = []
    
    for order in data_order:
        for path in meta_classifier_data_paths:
            if order in path:
                with open(path, "rb") as f:
                    all_data.append(pickle.load(f))
    
    if args.image_only:
        filepath = f"fine_tuned_models/{args.model_name}_image_only.hdf5"
        del all_data[-1]
    else:
        filepath = f"fine_tuned_models/{args.model_name}.hdf5"

    print("ALL DATA", len(all_data))
    dataset = all_data[0][0]
    for data in all_data[1:]:
        print("HI")
        print(data[0].shape)
        dataset = np.concatenate((dataset, data[0]), axis=1)
    
    # Take 90% of the predictions as training set
    x_train = dataset[int(len(dataset) * 0.1):]
    
    if validation_set:
        x_val = dataset[:int(len(dataset) * 0.1)]
        y_val = all_data[0][1][:int(len(dataset) * 0.1)]
    
    y_train = all_data[0][1][int(len(dataset) * 0.1):]

    
    model.add(layers.Dense(args.meta_classifier_num_dense_neurons, input_dim=dataset.shape[1], activation='relu'))
    model.add(layers.Dropout(args.meta_classifier_dropout))
    
    for _ in range(args.meta_classifier_hidden_layers):
        model.add(layers.Dense(args.meta_classifier_num_dense_neurons, activation='relu'))
        model.add(layers.Dropout(args.meta_classifier_dropout))
        
    model.add(layers.Dense(args.num_classes, activation='softmax'))
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.meta_classifier_learning_rate)

    # compile the model
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    if validation_set:
        cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        model.fit(x_train, np.array(y_train), validation_data=(x_val, np.array(y_val)), epochs=50, batch_size=32, callbacks=[es, cp])
    else:
        model.fit(x_train, np.array(y_train), epochs=10, batch_size=32)
        model.save(filepath)


def main():
    """The main function.
    Usage: python train_base.py -model_name -[model_name] -epochs [epochs] -image_only
    """
    
    parser = argparse.ArgumentParser("""For an image only model, train all except 'distilbert'. Start with the 'holistic' model, 
    then subsequently train the 'header', 'footer', 'left_body' and 'right_body'. Optionally,
    train 'distilbert', if -image_only is set to False. The default hyperparameters are set such that the highest test set accuracy on RVL-CDIP is achieved.\n""")

    # Required arguments
    parser.add_argument('-model_name', required=True, help="model_name is either 'holistic', 'header', 'footer', 'left_body', 'right_body' or 'distilbert'")
    parser.add_argument('-train_labels_paths', required=True, help="input the training paths in the same format as for RVL-CDIP (see e.g. data/labels/train.txt)")
    parser.add_argument('-num_classes', required=True, help="input the number of classes of the dataset")
    
    # Optional dataset arguments
    parser.add_argument('-val_labels_paths', help="input the validation paths in the same format as for RVL-CDIP (see e.g. data/labels/val.txt)")
    parser.add_argument('-min_dataset_size_for_val', default=10, help="the minimum size of the dataset per class in order to create a validation set if none is provided")
    
    # Optional model architecture argument
    parser.add_argument('-image_only', action=argparse.BooleanOptionalAction, default=False)

    # Optional image model hyperparameter arguments
    parser.add_argument('-image_epochs', default=20)   
    parser.add_argument('-image_epochs_if_no_val_set', default=10)
    parser.add_argument('-image_learning_rate', default=0.001)  
    parser.add_argument('-image_dropout', default=0.3)   
    parser.add_argument('-image_num_dense_neurons', default=50)
    parser.add_argument('-image_batch_size', default=32)   
    
    # Optional text model hyperparameter arguments
    parser.add_argument('-text_epochs', default=20)  
    parser.add_argument('-text_epochs_if_no_val_set', default=10) 
    parser.add_argument('-text_learning_rate', default=0.0005)  
    parser.add_argument('-text_dropout', default=0.3)   
    parser.add_argument('-text_num_dense_neurons', default=512)
    parser.add_argument('-text_batch_size', default=32)   
    
    # Optional metaclassifier hyperparameter arguments
    parser.add_argument('-meta_classifier_num_dense_neurons', default=256)
    parser.add_argument('-meta_classifier_learning_rate', default=0.001)
    parser.add_argument('-meta_classifier_dropout', default=0.3)
    parser.add_argument('-meta_classifier_hidden_layers', default=2)
     
    # Optional preprocessing hyperparameter arguments
    parser.add_argument('-image_shape', default=(384,384,3))
    parser.add_argument('-max_len', default=256)
     
    args = parser.parse_args()    
        
    assert args.model_name == "holistic" or args.model_name == "header" or args.model_name == "footer" or args.model_name == "left_body" or args.model_name == "right_body" or args.model_name == "distilbert" or args.model_name == "meta_classifier", "model_name must be set to one of ['holistic', 'header', 'footer', 'left_body', 'right_body', 'distilbert', 'meta_classifier']"   
    
    if args.model_name == "meta_classifier":
        assert len(os.listdir("fine_tuned_predictions")) == 5 or len(os.listdir("fine_tuned_predictions")) == 6, "first train all 5/6 base models (holistic, header, footer, left_body, right_body and optionally distilbert - depending on if an image only system is trained), then train the meta_classifier"
    
    if args.model_name == "distilbert":
        assert os.path.isdir('fine_tuned_data') and len(os.listdir('fine_tuned_data')) > 1, "extract the text first with extract_finetune_text.py"
    
    if args.model_name == "distilbert":
        train_text(args) 
    elif args.model_name == "meta_classifier":
        train_meta(args)
    else:
        train_image(args)
            
if __name__ == "__main__":
    main()
