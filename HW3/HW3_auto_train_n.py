#%%
from libsvm.svmutil import (
    svm_read_problem,
    svm_problem,
    svm_train,
    svm_save_model,
    svm_parameter,
    svm_predict,
)

import os
import pandas as pd
from copy import deepcopy
from rich import print as print
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re
import sys

GAMMER = 0.03125

FILE_TRAIN = "./train.txt"
FILE_TEST = "./test.txt"
TMP_FILE = "tmp.out"
MAIN_SAVE_FOLDER = "svm_s_auto_train"

#%%
PLATFORM_COMMAND = ".\windows\svm-predict.exe" if sys.platform == 'win32' else "./linux/svm-predict"


def predict_in_console(model_file_name, file_in, file_out):
    """
    This function takes in a model file, an input file, and an output file, runs a command using
    SVM-predict, and returns the accuracy of the prediction.
    
    :param model_file_name: The name of the file containing the trained SVM model
    :param file_in: The input file containing the data to be predicted
    :param file_out: The name of the file where the predicted values will be written
    :return: the accuracy of the prediction made by the SVM model.
    """

    cmd = f"{PLATFORM_COMMAND} {file_in} {model_file_name} {file_out}"
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0]
    output = output.decode('utf-8')
    print(output)

    regex = r'Accuracy = ([0-9\.]+)%'
    match = re.search(regex, output)

    return float(match.group(1)) if match else None


#%%
class svm_s_model_train_auto:

    def __init__(self):
        # self.save_folder = folder_name
        self.t_mode = list(range(4))
        self.t_mode_str = ["linear", "polynomial", "rbf", "sigmoid"]
        self.n = np.arange(0.1, 1.1, 0.1)

        self.g = GAMMER
        self.format_command = lambda n, g, t: f"-s 1 -n {n} -g {g} -t {t}"

        self.yx_train = svm_read_problem("train.txt")
        self.yx_test = svm_read_problem("test.txt")

        self.train_prob = svm_problem(self.yx_train[0], self.yx_train[1])
        self.test_prob = svm_problem(self.yx_test[0], self.yx_test[1])

        # save data frame
        self.df_train = pd.DataFrame(columns=self.t_mode_str)
        self.df_test = pd.DataFrame(columns=self.t_mode_str)

    def save_file_name(self, save_folder, n, t_mode):
        """
        This function returns a file name with a specific format based on the input parameters.
        
        :param save_folder: The directory path where the file will be saved
        :param n: The parameter "n" is a floating-point number that represents the value of the
        regularization parameter used in the SVM model
        :param t_mode: t_mode is a variable that represents the type of SVM model being used. It is an
        integer value that corresponds to a specific type of SVM model, and is used to generate a string
        representation of the model type in the saved file name
        :return: a string that represents the file path for saving a model. The file name includes the value
        of the variable `s` formatted as a float with one decimal place, the string "svm", and the value of
        the variable `t_mode` converted to a string using a dictionary `self.t_mode_str`. The file path is
        created using the `os.path.join()` function, which
        """

        return os.path.join(save_folder, f"model_n{n:.1f}_svm_{self.t_mode_str[t_mode]}")

    def train_term_model(self, n, t):
        """
        This function trains a support vector machine model with given parameters n and t.
        
        :param n: The parameter n is the degree of the polynomial kernel function used in the SVM algorithm.
        It is used to control the complexity of the decision boundary. A higher value of n will result in a
        more complex decision boundary, which may lead to overfitting if the training data is not
        representative of the underlying
        :param t: The parameter "t" in this function refers to the type of kernel function to be used in the
        SVM model. SVMs use kernel functions to transform the input data into a higher-dimensional space
        where it can be more easily separated into classes. The type of kernel function used can have a
        significant impact on
        :return: The function `train_term_model` returns an SVM model trained on the training data
        (`train_prob`) using the specified parameters (`n` and `t`).
        """

        print(n, t)
        param = svm_parameter(self.format_command(n=n, g=self.g, t=t))
        model = svm_train(self.train_prob, param)

        return model

    def predict_term(self, model_file_name):
        """
        This function predicts the accuracy of a model on training and testing data using a given model
        file.
        
        :param model_file_name: The name of the file containing the trained model that will be used for
        prediction
        :return: The function `predict_term` returns the accuracy of the model on the training and testing
        datasets. Specifically, it returns `train_p_acc` and `test_p_acc`, which are the prediction
        accuracies of the model on the training and testing datasets, respectively.
        """

        print(f"\n\nmodel_file_name : {model_file_name}")

        print("Train:")
        train_p_acc = predict_in_console(".\\" + model_file_name, FILE_TRAIN, TMP_FILE)
        print("p_acc: ", train_p_acc)

        print("Test:")
        test_p_acc = predict_in_console(".\\" + model_file_name, FILE_TEST, TMP_FILE)
        print("p_acc: ", test_p_acc)

        return train_p_acc, test_p_acc

    def save_model(self, save_folder, run_model_dict: dict) -> dict:

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        file_name_dict = {}

        for n, model_list in run_model_dict.items():

            file_name_list = []

            for t_mode_index, model in enumerate(model_list):
                file_name = self.save_file_name(save_folder=save_folder, n=n, t_mode=t_mode_index)
                svm_save_model(file_name, model)
                file_name_list.append(file_name)

            file_name_dict |= {n: deepcopy(file_name_list)}

        return file_name_dict

    def save_to_csv(self, csv_folder_path, model_predict_dict):
        """
        This function saves the training and testing accuracy of a model to CSV files in a specified folder
        path.
        
        :param csv_folder_path: The path to the folder where the CSV files will be saved
        :param model_predict_dict: A dictionary containing the model predictions for each class. The keys
        are the class names and the values are tuples containing the training accuracy and testing accuracy
        for that class
        :return: two pandas dataframes, `df_train` and `df_test`.
        """

        if not os.path.exists(csv_folder_path):
            os.makedirs(csv_folder_path)

        train_acc_dict = {c: pre[0] for c, pre, in model_predict_dict.items()}
        test_acc_dict = {c: pre[1] for c, pre, in model_predict_dict.items()}

        for c, train_acc in train_acc_dict.items():
            self.df_train.loc[str(c)] = dict(zip(self.t_mode_str, train_acc))

        for c, test_acc in test_acc_dict.items():
            self.df_test.loc[str(c)] = dict(zip(self.t_mode_str, test_acc))

        self.df_train.to_csv(os.path.join(csv_folder_path, f"svm_s_train.csv"))
        self.df_test.to_csv(os.path.join(csv_folder_path, f"svm_s_test.csv"))

        return self.df_train, self.df_test

    def save_plot_graph(self, folder_path):
        """
        This function saves bar plot graphs of the accuracy of a train and test model with different kernels
        in a specified folder path.
        
        :param folder_path: The path of the folder where the plot graphs will be saved
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        df_train = self.df_train.transpose()
        df_test = self.df_test.transpose()

        plt.clf()
        df_train.plot.bar(rot=0)

        plt.title("train model different kernel acc")
        plt.xlabel("kernel type")
        plt.ylabel("acc")

        plt.savefig(os.path.join(folder_path, "svm s train"))

        plt.clf()
        df_test.plot.bar(rot=0)

        plt.title("train model different kernel acc")
        plt.xlabel("kernel type")
        plt.ylabel("acc")

        plt.savefig(os.path.join(folder_path, "svm s test"))

    def save_support_vectors(self, folder_path: str, run_model_dict: dict):
        """
        This function saves the support vectors of a given model in a specified folder path and returns
        a dictionary containing the support vectors.
        
        :param folder_path: The path to the folder where the support vector files will be saved
        :type folder_path: str
        :param run_model_dict: The `run_model_dict` parameter is a dictionary where the keys are the
        values of the hyperparameter `C` used in training the SVM models, and the values are lists of
        trained SVM models for each value of `C`
        :type run_model_dict: dict
        :return: the dictionary `model_dict_sv`, which contains the support vector indices for each
        class and model. However, the function also saves the support vector lengths and support vectors
        themselves to files in the specified `folder_path`.
        """

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        model_dict_sv = {c: [model.get_sv_indices() for model in models] for c, models in run_model_dict.items()}
        model_dict_sv_len = {c: [len(sv) for sv in sv_len] for c, sv_len in model_dict_sv.items()}

        print(model_dict_sv_len)

        model_dict_sv_len = {
            f"{c:.1f}": dict(zip(self.t_mode_str, sv_len_list))
            for c, sv_len_list in model_dict_sv_len.items()
        }

        # to df
        df_model_sv_len = pd.DataFrame()
        for c, len_dict in model_dict_sv_len.items():
            df_model_sv_len[c] = len_dict

        df_model_sv_len.to_csv(os.path.join(folder_path, "support_vector_len.csv"))

        # plot
        plt.clf()
        df_model_sv_len_t = df_model_sv_len.transpose()
        df_model_sv_len_t.plot.bar(rot=0)

        plt.title("support vector number")
        plt.xlabel("s number")
        plt.ylabel("vector number")

        plt.savefig(os.path.join(folder_path, "support_vector_len"))

        for s, sv_list in model_dict_sv.items():
            if not os.path.exists(new_path := os.path.join(folder_path, f"{s:.1f}")):
                os.makedirs(new_path)

            for index, sv in enumerate(sv_list):
                using_type, sv_np = self.t_mode_str[index], np.array(sv)
                np.savetxt(os.path.join(new_path, f"svm_n{s:.1f}_{using_type}_support_vector.txt"), sv_np)

        return model_dict_sv

    def save_support_vector_coef(self, folder_path: str, run_model_dict: dict):
        """
        This function saves the support vector coefficients of a set of SVM models in a specified folder
        path.
        
        :param folder_path: The path to the folder where the support vector coefficients will be saved
        :type folder_path: str
        :param run_model_dict: run_model_dict is a dictionary containing the trained SVM models for
        different values of the regularization parameter (s). The keys of the dictionary are the values of
        s, and the values are lists of trained SVM models for that value of s
        :type run_model_dict: dict
        :return: a dictionary `model_dict_sv_coef` which contains the support vector coefficients for each
        model in `run_model_dict`. The keys of the dictionary are the values of `s` in `run_model_dict`, and
        the values are lists of numpy arrays containing the support vector coefficients for each model in
        the corresponding list in `run_model_dict`.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        model_dict_sv_coef = {s: [np.array(model.get_sv_coef()) for model in models] for s, models in run_model_dict.items()}

        for s, sv_coef_list in model_dict_sv_coef.items():
            if not os.path.exists(new_path := os.path.join(folder_path, str(f"{s:.1f}"))):
                os.makedirs(new_path)

            for index, sv_coef in enumerate(sv_coef_list):
                np.savetxt(os.path.join(new_path, f"svm_n{s:.1f}_{self.t_mode_str[index]}_coef.txt"), sv_coef)

        return model_dict_sv_coef

    def run(self):
        """
        The function runs a model for each value in a dictionary and returns the resulting model and
        prediction dictionaries.
        :return: two dictionaries: `run_model_dict` and `run_model_predict_dict`.
        """

        run_dict = {n: deepcopy(self.t_mode) for n in self.n}
        print(run_dict)
        # # save all model
        run_model_dict = {n: [self.train_term_model(n=n, t=t) for t in t_list] for n, t_list in run_dict.items()}

        model_file_dict = self.save_model(os.path.join(MAIN_SAVE_FOLDER, "model"), run_model_dict)

        run_model_predict_dict = None

        # ! look here
        run_model_predict_dict = {
            n: np.array([self.predict_term(model_file_name=model_file) for model_file in model_file_name_list]).T
            for n, model_file_name_list in deepcopy(model_file_dict).items()
        }

        if os.path.exists(TMP_FILE):
            os.remove(TMP_FILE)

        print(run_model_predict_dict)

        return run_model_dict, run_model_predict_dict


#%%
auto_train = svm_s_model_train_auto()
model_dict, model_predict_dict_acc = auto_train.run()
print(model_dict)

#%%
auto_train.save_support_vectors(os.path.join(MAIN_SAVE_FOLDER, "support vector"), model_dict)

#%%%
auto_train.save_support_vector_coef(os.path.join(MAIN_SAVE_FOLDER, "support vector coef"), model_dict)

#%%

auto_train.save_to_csv(os.path.join(MAIN_SAVE_FOLDER, "csv"), model_predict_dict_acc)

#%%
auto_train.save_plot_graph(os.path.join(MAIN_SAVE_FOLDER, "image"))
# %%