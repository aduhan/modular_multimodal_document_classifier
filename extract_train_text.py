import os
import json
import re

import pandas as pd
import pytesseract as py
from tqdm import tqdm

if not os.path.isdir('data/text'):
    os.mkdir("data/text")
        
base_path_images = "data/images/"
base_path_output_text = "data/text/"

train_path = "data/labels/train.txt"
val_path = "data/labels/val.txt"
test_path = "data/labels/test.txt"

train_paths = pd.read_csv(train_path, header=None)[0].apply(lambda x: x.split(" "))
val_paths = pd.read_csv(val_path, header=None)[0].apply(lambda x: x.split(" "))
test_paths = pd.read_csv(test_path, header=None)[0].apply(lambda x: x.split(" "))

# the following file from the test set was found to be corrupt
corrupt_file_path = "imagese/e/j/e/eje42e00/2500126531_2500126536.tif"
test_paths = test_paths[test_paths.apply(lambda x: corrupt_file_path not in x)].reset_index(drop=True)

all_paths = [train_paths, val_paths, test_paths]
data_parts = ["train", "val", "test"]

for data_part, paths in zip(data_parts, all_paths):

    dataset = {}
    for i, file_path in tqdm(enumerate(paths), total=len(paths)):
        text_data = py.image_to_data(base_path_images + file_path[0], output_type="data.frame")[["text"]]
        try:
            text_data = py.image_to_data(base_path_images + file_path[0], output_type="data.frame")[["text"]]
            
            # Do some preprocessing
            text_data = text_data[~pd.isnull(text_data.text)].reset_index(drop=True) # ignore rows, where text = NaN, i.e. Boxes, which contain more than 1 word
            text_data.text = text_data.text.apply(lambda x: re.sub(r"[^a-zA-Z0-9]","", x).lower()) # keep just letters and digits and make all words lowercase
            text_data = text_data[text_data.text.apply(lambda x: not str.isspace(str(x)))] # ignore rows, where text is just whitespace
            text_data = text_data[text_data.text.apply(lambda x: len(str(x)) > 1 or x.isdigit())] # ignore rows, where text is 0 or 1 character long    
            dataset.update({file_path[0]: " ".join(list(text_data.text))})
            
        except:
            dataset.update({file_path[0]: ""})
            continue

    with open(base_path_output_text + data_part + ".json", "w") as file:
        json.dump(dataset, file)