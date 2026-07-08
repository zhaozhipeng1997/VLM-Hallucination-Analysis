import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image




class POPEDataSet(Dataset):
    def __init__(self, pope_path, data_path):
        self.pope_path = pope_path
        self.data_path = data_path

        image_list, query_list, label_list, labels = [], [], [], []

        for q in open(pope_path, 'r'):
            line = json.loads(q)
            image_list.append(line['image'])
            query_list.append(line['text'])
            label_list.append(line['label'])
            labels.append(line['label'])

        for i in range(len(label_list)):
            if label_list[i] == 'no' or label_list[i] == 'No' or label_list[i] == 'Not' or label_list[i] == 'NO':
                label_list[i] = 0
            else:
                label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list
        self.labels = labels

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        
        # raw_image = Image.open(image_path).convert("RGB")
        # image = raw_image
        query = self.query_list[index]
        label = self.label_list[index]
        label_txt = self.labels[index]

        return {"image_path":image_path, "query": query, "label": label, "label_txt": label_txt}