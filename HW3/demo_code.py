import subprocess
import re
import os
import sys

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


print(predict_in_console('./svm_s_auto_train/model/model_s0.1_svm_linear', "./train.txt", "./tmp.txt"))

if os.path.exists("./tmp.txt"):
    os.remove("./tmp.txt")
