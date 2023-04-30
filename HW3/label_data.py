from PIL import Image
import numpy as np
from collections import Counter
from rich import print as print
import os, sys , re


CLASS_NUMBER = 5

def extract_features(file_name):
    """
    The function takes in an image file name, converts it to grayscale, resizes it to 28x28, normalizes
    the pixel values to [0,1], and returns a string in libsvm format representing the pixel values.
    
    :param file_name: The file name (including the path) of the image file that needs to be processed
    :return: a string in libsvm format that represents the features extracted from an image file. The
    string contains the index of each feature and its corresponding value, separated by a colon and
    formatted to 6 decimal places.
    """
    img = Image.open(file_name)

    img = img.convert("L")

    img = img.resize((28, 28))

    pixels = np.array(list(img.getdata()))

    # 将像素值标准化为 [0, 1] 的范围
    pixels = pixels / 255.0

    # 将像素值格式化为 libsvm 格式的字符串
    return " ".join([f"{i + 1}:{p:.6e}" for i, p in enumerate(pixels)])

def get_folder_list(folder_path):
    
    def sort_file_name_key(file_name):
        match_para = re.search(r"\d+" , file_name)
        
        return int(match_para.group()) if match_para else file_name
    
    if not os.path.exists(folder_path):
        print(f"error : folder path {folder_path} is not find") 
        
        return None
    
    folder_file_name = sorted(os.listdir(folder_path) , key=sort_file_name_key)
    
    return [os.path.join(folder_path, file_name) for file_name in folder_file_name]

def label_data(folder_data_list):
    label_number_list = [i for i in range(CLASS_NUMBER) for _ in range(len(folder_data_list) // CLASS_NUMBER)]
    
    input_format = [extract_features(file_name) for file_name in folder_data_list]
    label_list_it = [f"{i} {val}" for i ,val in zip(label_number_list, input_format)]
    
    return label_list_it

def main() ->None:
    
    folder_train_path , folder_test_path = "./train" , "./test"
    
    folder_train_file_list = get_folder_list(folder_path=folder_train_path)
    folder_test_file_list = get_folder_list(folder_path=folder_test_path)
    
    folder_train_file_len = len(folder_train_file_list)
    folder_test_file_len = len(folder_test_file_list)
    
    print(f"train data size {folder_train_file_len}")
    print(f"test data size {folder_test_file_len}")
    
    train_data_label_list = label_data(folder_train_file_list)
    test_data_label_list = label_data(folder_test_file_list)
    
    # get the svm input format
    
    
    with open("train.txt", "w") as f:
        f.write('\n'.join(train_data_label_list))

    with open("test.txt", "w") as f:
        f.write('\n'.join(test_data_label_list))
    
    


if __name__ == "__main__":
    main()


