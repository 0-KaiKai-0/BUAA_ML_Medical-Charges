import pandas as pd
import numpy as np


class data_loader():
    def __init__(self, root):
        self.root = root
        self.data = pd.read_csv(root)
        self.target = self.data.columns[-1]

    def preprocess(self):
        self.data.loc[self.data['sex'] == 'male', 'sex'] = 0
        self.data.loc[self.data['sex'] == 'female', 'sex'] = 1
        self.data.loc[self.data['smoker'] == 'no', 'smoker'] = 0
        self.data.loc[self.data['smoker'] == 'yes', 'smoker'] = 1
        self.data.loc[self.data['region'] == 'southwest', 'region'] = 0
        self.data.loc[self.data['region'] == 'southeast', 'region'] = 1
        self.data.loc[self.data['region'] == 'northeast', 'region'] = 2
        self.data.loc[self.data['region'] == 'northwest', 'region'] = 3

    def normalize(self, mean, std):
        for column in self.data.columns:
            if column == self.target:
                continue
            self.data[column] = \
                (self.data[column] - mean[column]) / std[column]

    def forward(self, is_train=True, mean=None, std=None):
        self.preprocess()
        if is_train:
            mean, std = {}, {}
            for column in self.data.columns:
                if column == self.target:
                    continue
                mean[column] = self.data[column].mean()
                std[column] = self.data[column].std()
        self.normalize(mean, std)
        return mean, std
