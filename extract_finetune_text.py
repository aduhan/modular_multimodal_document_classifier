import os
import json
import argparse
import re

import pandas as pd
import pytesseract as py
from tqdm import tqdm
from sklearn.model_selection import train_test_split 

if not os.path.isdir('fine_tuned_data'):
    os.mkdir("fine_tuned_data")

def get_text(args: argparse.Namespace):
    """Extract the text from the images.
    
    Args:
        args: the arguments input
    """
        
    train_paths = pd.read_csv(args.train_labels_paths, header=None)[0].apply(lambda x: x.split(" "))
    train_paths = train_paths.sample(frac=1).reset_index(drop=True)
    
    all_paths = [train_paths]
    data_parts = ["train"]
    
    if not isinstance(args.val_labels_paths, type(None)):
        val_paths_df = pd.read_csv(args.val_labels_paths, header=None).sample(frac=1).reset_index(drop=True)
        val_paths = val_paths_df[0].apply(lambda x: x.split(" "))
        all_paths.append(val_paths)
        data_parts.append("val")    
        
    # if no path for validation labels is given and the number of examples per class is greater than min_dataset_size_for_val
    elif isinstance(args.val_labels_paths, type(None)) and (pd.DataFrame.from_records([row[1] for row in train_paths]).value_counts() > int(args.min_dataset_size_for_val)).all():
        temp_df = pd.DataFrame.from_records([(row[0], row[1]) for row in train_paths])
        train_paths, val_paths = train_test_split(temp_df, test_size=0.1, stratify=temp_df[1], random_state=1) 
        
        train_paths = pd.Series(train_paths[0] + " " + train_paths[1]).apply(lambda x: x.split(" "))
        val_paths = pd.Series(val_paths[0] + " " + val_paths[1]).apply(lambda x: x.split(" "))
        
        all_paths = [train_paths, val_paths]
        data_parts.append("val")  
    
    for data_part, paths in zip(data_parts, all_paths):

        dataset = {}
        labels = {}
        for i, file_path in tqdm(enumerate(paths), total=len(paths)):
            text_data = py.image_to_data(file_path[0], output_type="data.frame")[["text"]]
            try:
                text_data = py.image_to_data(file_path[0], output_type="data.frame")[["text"]]
                
                # Do some preprocessing
                text_data = text_data[~pd.isnull(text_data.text)].reset_index(drop=True) # ignore rows, where text = NaN, i.e. Boxes, which contain more than 1 word
                text_data.text = text_data.text.apply(lambda x: re.sub(r"[^a-zA-Z0-9]","", x).lower()) # keep just letters and digits and make all words lowercase
                text_data = text_data[text_data.text.apply(lambda x: not str.isspace(str(x)))] # ignore rows, where text is just whitespace
                text_data = text_data[text_data.text.apply(lambda x: len(str(x)) > 1 or x.isdigit())] # ignore rows, where text is 0 or 1 character long    
                dataset.update({file_path[0]: " ".join(list(text_data.text))})
                labels.update({file_path[0]:file_path[1]})
                
            except:
                dataset.update({file_path[0]: ""})
                labels.update({file_path[0]:file_path[1]})
                continue

        with open("fine_tuned_data/" + data_part + ".json", "w") as file:
            json.dump(dataset, file)
            
        with open("fine_tuned_data/" + data_part + "_labels.json", "w") as file:
            json.dump(labels, file)
            
            

def main():
    """The main function.
    Usage: python train_base.py -model_name -[model_name] -epochs [epochs] -image_only
    """
    
    parser = argparse.ArgumentParser()
    
    # Required argument
    parser.add_argument('-train_labels_paths', required=True, help="input the training paths in the same format as for RVL-CDIP (see e.g. data/labels/train.txt)")
    
    # Optional arguments
    parser.add_argument('-val_labels_paths', help="input the validation paths in the same format as for RVL-CDIP (see e.g. data/labels/val.txt)")
    parser.add_argument('-min_dataset_size_for_val', default=10, help="the minimum size of the dataset per class in order to create a validation set if none is provided")
    
    args = parser.parse_args()
    
    get_text(args)
            
if __name__ == "__main__":
    main()
