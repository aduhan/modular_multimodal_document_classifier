import json
from typing import Tuple

import pandas as pd


def get_paths(model_name: str) -> Tuple:
    """Get the paths for train, val and test set.
    
    Args:
        model_name: name of the model
        
    Returns:
        Tuple: a tuple of different file paths
    """
    
    if model_name != "distilbert":
        train_path = "data/labels/train.txt"
        val_path = "data/labels/val.txt"
        test_path = "data/labels/test.txt"

        train_paths = pd.read_csv(train_path, header=None)[0].apply(lambda x: x.split(" "))
        val_paths = pd.read_csv(val_path, header=None)[0].apply(lambda x: x.split(" "))
        test_paths = pd.read_csv(test_path, header=None)[0].apply(lambda x: x.split(" "))
        
        y_val = list(pd.read_csv(val_path, header=None)[0].apply(lambda x: int(x.split(" ")[1])))

        # the following file from the test set was found to be corrupt
        corrupt_file_path = "imagese/e/j/e/eje42e00/2500126531_2500126536.tif"
        test_paths = test_paths[test_paths.apply(lambda x: corrupt_file_path not in x)].reset_index(drop=True)
        y_test = list(test_paths.apply(lambda x: int(x[1])))       
        
        return train_paths, val_paths, test_paths, y_val, y_test
    
    else:
        
        train_text_path = "data/text/train.json"
        val_text_path = "data/text/val.json"
        test_text_path = "data/text/test.json"
        
        with open(train_text_path, "r") as f:
            train_text = json.load(f)

        with open(val_text_path, "r") as f:
            val_text = json.load(f)

        with open(test_text_path, "r") as f:
            test_text = json.load(f)

        X_train = list(train_text.values())
        X_val = list(val_text.values())
        X_test = list(test_text.values())
        
        train_label_path = "data/labels/train.txt"
        val_label_path = "data/labels/val.txt"
        test_label_path = "data/labels/test.txt"
        
        y_train = list(pd.read_csv(train_label_path, header=None)[0].apply(lambda x: int(x.split(" ")[1])))
        y_val = list(pd.read_csv(val_label_path, header=None)[0].apply(lambda x: int(x.split(" ")[1])))
        test_paths = pd.read_csv(test_label_path, header=None)[0].apply(lambda x: x.split(" "))
        
        # the following file from the test set was found to be corrupt
        corrupt_file_path = "imagese/e/j/e/eje42e00/2500126531_2500126536.tif"
        test_paths = test_paths[test_paths.apply(lambda x: corrupt_file_path not in x)].reset_index(drop=True)
        y_test = list(test_paths.apply(lambda x: int(x[1])))
        
        return X_train, X_val, X_test, y_train, y_val, y_test
