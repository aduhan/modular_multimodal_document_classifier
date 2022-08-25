import argparse
import os
import re
from glob import glob
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import pytesseract as py
from PIL import Image
from tensorflow.keras.models import load_model
from transformers import DistilBertTokenizer, TFDistilBertModel


def transform_image(file_path: str, model_name: str, image_shape: Tuple) -> np.array:
    """Transfrom an image according to the model

    Args:
        file_path: path of a file
        model_name: name of the model
        image_shape: shape of the resized image
        
    Returns:
        np.ndarray: a 3-dimensional normalized array
    """
        
    holistic = np.asarray(Image.open(file_path).resize((720, 936), Image.ANTIALIAS))/255
    top_to_body_pixels = int((holistic.shape[0]-480)/2)

    if model_name == "holistic":
        holistic = cv2.resize(holistic, image_shape[:2])
        holistic = np.repeat(holistic[np.newaxis, :, :, np.newaxis], 3, axis=3)
        
        return holistic

    elif model_name == "header":
        header = holistic[:307,:]
        header = cv2.resize(header, image_shape[:2])
        header = np.repeat(header[np.newaxis, :,:, np.newaxis], 3, axis=3)
        
        return header

    elif model_name == "footer":
        footer = holistic[holistic.shape[0]-307:,:]
        footer = cv2.resize(footer, image_shape[:2])
        footer = np.repeat(footer[np.newaxis, :,:, np.newaxis], 3, axis=3)
        
        return footer
    
    elif model_name == "left_body":
        left_body = holistic[top_to_body_pixels:holistic.shape[0]-top_to_body_pixels, :360]
        left_body = cv2.resize(left_body, image_shape[:2])
        left_body = np.repeat(left_body[np.newaxis, :,:, np.newaxis], 3, axis=3)
        
        return left_body

    elif model_name == "right_body":
        right_body = holistic[top_to_body_pixels:holistic.shape[0]-top_to_body_pixels, 360:]
        right_body = cv2.resize(right_body, image_shape[:2])
        right_body = np.repeat(right_body[np.newaxis, :,:, np.newaxis], 3, axis=3)
        
        return right_body

def transform_text(file_path: str, max_len: int) -> Tuple[np.array, np.array]:
    """Extract the text of an image and preprocess it.
    
    Args:
        file_path: path of a file.
        max_len: maximum length for a document representation
        
    Returns:
        input: an array of string to id mappings
        attention_mask: an array of zeros and ones
    """
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    try:
        text_data = py.image_to_data(file_path, output_type="data.frame")[["text"]]

        # Do some preprocessing
        text_data = text_data[~pd.isnull(text_data.text)].reset_index(drop=True) # ignore rows, where text = NaN, i.e. Boxes, which contain more than 1 word
        text_data.text = text_data.text.apply(lambda x: re.sub(r"[^a-zA-Z0-9]","", x).lower()) # keep just letters and digits and make all words lowercase
        text_data = text_data[text_data.text.apply(lambda x: not str.isspace(str(x)))] # ignore rows, where text is just whitespace
        text_data = text_data[text_data.text.apply(lambda x: len(str(x)) > 1 or x.isdigit())] # ignore rows, where text is 0 or 1 character long 
                   
    except:
        
        text_data = [""]

    input = []
    attention_masks = []
    
    for x in text_data:
        train_dbert_inps=tokenizer.encode_plus(x, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, return_attention_mask=True, truncation=True)
        input.append(train_dbert_inps['input_ids'])
        attention_masks.append(train_dbert_inps['attention_mask'])  
    
    input = np.array(input)
    attention_masks = np.array(attention_masks)  
    
    return input, attention_masks
    

def test(args: argparse.Namespace):
    """Test the model in either image_only mode
    or multimodal mode.
    
    Args:
        args: the arguments input
    """
    
    prediction_mapping = {0: "Letter",
                          1: "Form",
                          2: "Email",
                          3: "Handwritten",
                          4: "Advertisement",
                          5: "Scientific report",
                          6: "Scientific publication",
                          7: "Specification",
                          8: "File folder",
                          9: "News article",
                          10: "Budget",
                          11: "Invoice",
                          12: "Presentation",
                          13: "Questionnaire",
                          14: "Resume",
                          15: "Memo"
                          }
    
    predictions = []
    
    model_paths = glob("models/*")
    model_names = ["holistic", "header", "footer", "left_body", "right_body", "distilbert"]
    
    for model_name in model_names:
        for model_path in model_paths:
            if model_name in model_path:
                if args.image_only:
                    if "distilbert" in model_path:
                        continue
                    
                    input_data = transform_image(args.file_path, model_name, args.image_shape)
                    model = load_model(model_path)
                    prediction = model.predict(input_data)
                    predictions.append(prediction)
                        
                else:
                    
                    if "distilbert" in model_path:
                        model = load_model(model_path, custom_objects={"CustomModel": TFDistilBertModel})
                        input_data = transform_text(args.file_path, args.max_len)
                    else:
                        model = load_model(model_path)
                        input_data = transform_image(args.file_path, model_name, args.image_shape)
                        
                    prediction = model.predict(input_data)
                    predictions.append(prediction)
                    
                    
    dataset = predictions[0]
    for data in predictions[1:]:
        dataset = np.concatenate((dataset, data), axis=1)
    
    if args.image_only:
        meta_classifier_model = load_model("models/meta_classifier_image_only.hdf5")
    else:
        meta_classifier_model = load_model("models/meta_classifier.hdf5")
        
    prediction = meta_classifier_model.predict(dataset)
    
    print("Predicted category:", prediction_mapping[np.argmax(prediction)])    
            
                    
def main():
    """The main function.
    Usage: python train_base.py -file_path [file_path] [-image_only]
    """
    
    parser = argparse.ArgumentParser("Input the absolute/relative file path to get the prediction. Optionally, specify whether to use an image only system with -image_only.\n")
  
    # Required argument
    parser.add_argument('-file_path', required=True)
    
    # Optional model architecture argument
    parser.add_argument('-image_only', action=argparse.BooleanOptionalAction, default=False)

    # Optional preprocessing hyperparameter arguments. 
    # Have to be the same values as in the training script.
    parser.add_argument('-image_shape', default=(384,384,3))
    parser.add_argument('-max_len', default=256)

    args = parser.parse_args()
    
    if args.image_only:
        assert "meta_classifier_image_only.hdf5" in os.listdir("models/"), "First train an image only meta classifier model"
        
    test(args)
    
if __name__ == "__main__":
    main()
