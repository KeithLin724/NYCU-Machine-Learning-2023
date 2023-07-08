import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
from natsort import natsorted


class DemoClass:

    def __init__(self, folder_path, model, transform) -> None:
        self.folder_path = folder_path
        self.model = model
        self.transform = transform
        self.init_classes()
        self.load_sample_list()

    def load_sample_list(self):
        self.sample_list = os.listdir(self.folder_path)[1:]
        self.sample_list = natsorted(self.sample_list)
        self.sample_list = [os.path.join(self.folder_path, name) for name in self.sample_list]

    def init_classes(self):
        self.CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spyder', 'squirrel']
        self.RE_CLASSES = [
            'dog',
            'cat',
            'horse',
            'spyder',
            'butterfly',
            'chicken',
            'sheep',
            'cow',
            'squirrel',
            'elephant',
        ]

        self.RE_MAP = dict(zip(self.RE_CLASSES, range(len(self.RE_CLASSES))))

    def to_model_input_format(self, image):
        img_tensor = self.transform(image)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def run(self, device):
        save_result = []
        for image_path in tqdm(self.sample_list, unit="picture"):
            img = Image.open(image_path).convert("RGB")

            img_tensor = self.to_model_input_format(img)
            img_tensor = img_tensor.to(device)

            pred_prob = self.model(img_tensor)
            pred = torch.max(pred_prob, 1).indices
            pred = pred.item()

            save_result.append(self.CLASSES[pred])

        save_result_re_map = list(map(self.RE_MAP.get, save_result))
        return pd.Series(save_result_re_map)

    @staticmethod
    def df_to_excel(df, file_path: str):

        file_path = file_path if file_path.endswith(".xlsx") else f"{file_path}.xlsx"

        df.to_excel(file_path, header=False, index=False)
        return
