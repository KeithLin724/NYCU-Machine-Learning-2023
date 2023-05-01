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

GAMMER = 0.03125

MAIN_FOLDER = "svm_c_auto_train"


class svm_c_model_train_auto:

    def __init__(self):
        # self.save_folder = folder_name
        self.t_mode = list(range(4))
        self.t_mode_str = ["linear", "polynomial", "rbf", "sigmoid"]
        self.c = [2, 8]
        self.g = GAMMER
        self.format_command = lambda c, g, t: f"-c {c} -g {g} -t {t}"

        self.yx_train = svm_read_problem("train.txt")
        self.yx_test = svm_read_problem("test.txt")

        self.train_prob = svm_problem(self.yx_train[0], self.yx_train[1])
        self.test_prob = svm_problem(self.yx_test[0], self.yx_test[1])

        # save data frame
        self.df_train = pd.DataFrame(columns=self.t_mode_str)
        self.df_test = pd.DataFrame(columns=self.t_mode_str)

    def save_file_name(self, save_folder, c, t_mode):
        """
        This function returns a file path for saving a model with a specific name format based on input
        parameters.
        
        :param save_folder: The directory path where the file will be saved
        :param c: The parameter "c" is likely a hyperparameter used in a support vector machine (SVM) model.
        It controls the trade-off between achieving a low training error and a low testing error. A higher
        value of "c" will result in a more complex model that fits the training data more closely
        :param t_mode: t_mode is a variable that represents the type of data transformation mode being used.
        It is likely an integer or string value that corresponds to a specific data transformation method,
        such as normalization or standardization. The function uses this variable to generate a string that
        is included in the file name when saving a model
        :return: a string that represents the file path for saving a model. The file path is created using
        the `os.path.join()` function to join the `save_folder` parameter (which is a string representing
        the directory where the file will be saved), the string
        `"model_c{c}_svm_{self.t_mode_str[t_mode]}"`, where `c` and `t_mode` are
        """
        return os.path.join(save_folder, f"model_c{c}_svm_{self.t_mode_str[t_mode]}")

    def train_term_model(self, c, t):
        """
        This function trains a support vector machine model with given parameters and returns the trained
        model.
        
        :param c: The parameter C is a regularization parameter in SVM (Support Vector Machine) that
        controls the trade-off between achieving a low training error and a low testing error. A smaller
        value of C creates a wider margin, allowing more errors in the training data, while a larger value
        of C creates a narrower margin,
        :param t: The parameter "t" in this function refers to the type of kernel function to be used in the
        SVM model. The kernel function is used to transform the input data into a higher-dimensional space
        where it can be more easily separated by a hyperplane. There are several types of kernel functions
        available in SVM
        :return: The function `train_term_model` returns a trained SVM model.
        """
        print(c, t)
        param = svm_parameter(self.format_command(c=c, g=self.g, t=t))
        model = svm_train(self.train_prob, param)

        return model

    def predict_term(self, model, c, t):
        """
        This function predicts the accuracy of a given SVM model on training and testing data.
        
        :param model: The trained SVM model that will be used for prediction
        :param c: The regularization parameter in SVM, which controls the trade-off between achieving a low
        training error and a low testing error. A smaller value of c will result in a wider margin and more
        training errors, while a larger value of c will result in a narrower margin and fewer training
        errors
        :param t: The parameter "t" represents the kernel function used in the SVM model. It is a string
        that specifies the type of kernel function to be used, such as "linear", "polynomial", "radial basis
        function (RBF)", etc
        :return: the training accuracy and testing accuracy of a given SVM model, using the specified kernel
        function and parameters.
        """
        print(f"\n\nmodel c={c}, gammer={self.g}, kernel={self.t_mode_str[t]}")
        print("Train:")
        _, train_p_acc, _ = svm_predict(self.yx_train[0], self.yx_train[1], model)
        print("p_acc: ", train_p_acc)  # 预测的精确度，均值和回归的平方相关系数

        print("Test:")
        _, test_p_acc, _ = svm_predict(self.yx_test[0], self.yx_test[1], model)
        print("p_acc: ", test_p_acc)

        train_only_acc, _, _ = train_p_acc
        test_only_acc, _, _ = test_p_acc

        return train_only_acc, test_only_acc

    def save_model(self, save_folder, run_model_dict: dict):
        """
        This function saves SVM models to a specified folder.
        
        :param save_folder: The directory path where the model will be saved
        :param run_model_dict: `run_model_dict` is a dictionary containing the models to be saved. The keys
        of the dictionary represent the different values of a hyperparameter `c`, and the values are lists
        of models trained with that value of `c`. The `save_model` function saves each model in the
        appropriate file,
        :type run_model_dict: dict
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for c, model_list in run_model_dict.items():

            for t_mode_index, model in enumerate(model_list):
                svm_save_model(self.save_file_name(save_folder=save_folder, c=c, t_mode=t_mode_index), model)

    def save_to_csv(self, csv_folder_path, model_predict_dict):
        """
        This function saves the training and testing accuracy of a model to CSV files in a specified folder
        path.
        
        :param csv_folder_path: The path to the folder where the CSV files will be saved
        :param model_predict_dict: model_predict_dict is a dictionary containing the predicted accuracy
        values for each class label for both the training and testing datasets. The keys of the dictionary
        are the class labels and the values are tuples containing the predicted accuracy for the training
        and testing datasets respectively
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

        self.df_train.to_csv(os.path.join(csv_folder_path, f"svm_c_train.csv"))
        self.df_test.to_csv(os.path.join(csv_folder_path, f"svm_c_test.csv"))

        return self.df_train, self.df_test

    def save_plot_graph(self, folder_path):
        """
        This function saves bar plot graphs of the accuracy of a train and test model with different kernel
        values.
        
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

        plt.savefig(os.path.join(folder_path, "svm c train"))

        plt.clf()
        df_test.plot.bar(rot=0)

        plt.title("test model different kernel acc")
        plt.xlabel("kernel type")
        plt.ylabel("acc")

        plt.savefig(os.path.join(folder_path, "svm c test"))

    def model_SV_to_string(self, model_sv_list: list) -> str:
        """
        This function converts a list of support vector machine model support vectors to a string format.
        
        :param model_sv_list: A list of dictionaries representing the support vectors of a machine learning
        model. Each dictionary contains the feature index as the key and the corresponding feature value as
        the value
        :type model_sv_list: list
        :return: The function `model_SV_to_string` takes a list of dictionaries as input and returns a
        string. The string is created by applying the `sv_line_func` function to each dictionary in the
        input list and joining the resulting strings with newline characters. The `sv_line_func` function
        converts each dictionary into a string by joining the key-value pairs with colons and the items with
        spaces.
        """

        def sv_line_func(sv_item) -> str:
            return " ".join(":".join(map(str, item)) for item in sv_item.items())

        return "\n".join(sv_line_func(sv_item) for sv_item in model_sv_list)

    def save_support_vectors(self, folder_path: str, run_model_dict: dict):
        """
        This function saves the support vectors of a trained SVM model to a specified folder path and
        returns a dictionary containing the support vectors.
        
        :param folder_path: The path to the folder where the support vector files will be saved
        :type folder_path: str
        :param run_model_dict: The `run_model_dict` parameter is a dictionary where the keys are values
        of the hyperparameter `C` and the values are lists of trained SVM models for each combination of
        hyperparameters
        :type run_model_dict: dict
        :return: the dictionary `model_dict_sv`, which contains the support vector indices for each
        class and each model. However, the function is also saving the support vector lengths to a CSV
        file and a bar plot of the support vector lengths to a PNG file, and it is saving the support
        vectors themselves to text files.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        model_dict_sv = {c: [model.get_SV() for model in models] for c, models in run_model_dict.items()}
        model_dict_sv_len = {c: [len(sv) for sv in sv_len] for c, sv_len in model_dict_sv.items()}

        model_dict_sv_len = {str(c): dict(zip(self.t_mode_str, sv_len_list)) for c, sv_len_list in model_dict_sv_len.items()}

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
        plt.xlabel("c number")
        plt.ylabel("vector number")

        plt.savefig(os.path.join(folder_path, "support_vector_len"))

        for c, sv_list in model_dict_sv.items():
            if not os.path.exists(new_path := os.path.join(folder_path, str(c))):
                os.makedirs(new_path)

            for index, sv in enumerate(sv_list):
                using_type, sv_str = self.t_mode_str[index], self.model_SV_to_string(model_sv_list=sv)

                save_model_new_path = os.path.join(new_path, f"svm_c{c}_{using_type}_support_vector.txt")

                # to file
                with open(save_model_new_path, "w") as f:
                    f.write(sv_str)

        return model_dict_sv

    def save_support_vector_coef(self, folder_path: str, run_model_dict: dict):
        """
        This function saves the support vector coefficients of a set of SVM models in a specified folder
        path.
        
        :param folder_path: A string representing the path to the folder where the support vector
        coefficients will be saved
        :type folder_path: str
        :param run_model_dict: run_model_dict is a dictionary where the keys are the values of the
        hyperparameter C used in the SVM model and the values are lists of trained SVM models for each time
        mode
        :type run_model_dict: dict
        :return: a dictionary `model_dict_sv_coef` which contains the support vector coefficients for each
        class `c` in the `run_model_dict`. The support vector coefficients are obtained from the
        `get_sv_coef()` method of each model in the `run_model_dict`. The coefficients are saved in text
        files in the specified `folder_path` for each class and each mode (train or test).
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        model_dict_sv_coef = {c: [np.array(model.get_sv_coef()) for model in models] for c, models in run_model_dict.items()}

        for c, sv_coef_list in model_dict_sv_coef.items():
            if not os.path.exists(new_path := os.path.join(folder_path, str(c))):
                os.makedirs(new_path)

            for index, sv_coef in enumerate(sv_coef_list):
                np.savetxt(os.path.join(new_path, f"svm_c{c}_{self.t_mode_str[index]}_coef.txt"), sv_coef)

        return model_dict_sv_coef

    def run(self):
        """
        The function runs a model for each value in a dictionary and returns the resulting model and
        prediction dictionaries.
        :return: two dictionaries: `run_model_dict` and `run_model_predict_dict`.
        """

        run_dict = {c: deepcopy(self.t_mode) for c in self.c}
        print(run_dict)
        # # save all model
        run_model_dict = {c: [self.train_term_model(c=c, t=t) for t in t_list] for c, t_list in run_dict.items()}

        run_model_predict_dict = {
            c:
            np.array([self.predict_term(model=model, c=c, t=self.t_mode[t_index])
                      for t_index, model in enumerate(model_list)]).T
            for c, model_list in run_model_dict.items()
        }

        print(run_model_predict_dict)

        return run_model_dict, run_model_predict_dict


#%%
auto_train = svm_c_model_train_auto()
model_dict, model_predict_dict_acc = auto_train.run()

# %%
auto_train.save_model(os.path.join(MAIN_FOLDER, "model"), model_dict)

#%%
auto_train.save_to_csv(os.path.join(MAIN_FOLDER, "csv"), model_predict_dict_acc)

#%%
auto_train.save_plot_graph(os.path.join(MAIN_FOLDER, "image"))

# %%
auto_train.save_support_vectors(os.path.join(MAIN_FOLDER, "support vector"), model_dict)

# %%
auto_train.save_support_vector_coef(os.path.join(MAIN_FOLDER, "support vector coef"), model_dict)

# %%