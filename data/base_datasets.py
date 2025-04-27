import contextlib
import io
import json
import logging
import os.path
import random
import re
import time

import pandas as pd

from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX

from data.process_text import load_and_transform_text
from data.process_touch import load_and_transform_touch, get_touch_transform
from data.process_vision import load_and_transform_vision, get_vision_transform



import argparse
from os.path import join as opj
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import torch
from PIL import Image
from torchvision import transforms
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


SSVTP_dir = 'tactile_datasets/train/ssvtp/'
TAG_dir = 'tactile_datasets/train/TAG/dataset/'
obreal_dir = 'tactile_datasets/objectfolder/real/tactile/'
visgel_dir = 'tactile_datasets/visgel/images/touch/'
yuan18_dir = 'tactile_datasets/yuan18/Data_ICRA18/Data/'
TVL_dir = 'tactile_datasets/TVL/tvl_dataset/hct/'
ycb_dir = 'tactile_datasets/YCB-Slide/real/'
octopi_dir = 'tactile_datasets/octopi/'
text_dir = 'tactile_datasets/text/'

TAG_file = 'tactile_datasets/contact_text_tag_notest.csv'
obreal_file = 'tactile_datasets/contact_text_obj.csv'
visgel_file = 'tactile_datasets/contact_visgel.csv'
yuan18_file = 'tactile_datasets/contact_yuan.csv'
octopi_file = 'tactile_datasets/contact_text_octopi.csv'
TVL_file = 'tactile_datasets/contact_text_tvl.csv'

tacquad_indoor_dir = 'tactile_datasets/tacquad/data_indoor/'
tacquad_outdoor_dir = 'tactile_datasets/tacquad/data_outdoor/'

tacquad_indoor_file = 'tactile_datasets/tacquad/contact_indoor.csv'
tacquad_outdoor_file = 'tactile_datasets/tacquad/contact_outdoor.csv'

tacquad_text_dir = 'tactile_datasets/text_tacquad/'


class TOUCH_dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        
        self.touch_list = []
        self.vision_list = []
        self.text_list = []

        with open(TAG_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                folder = row[0]
                image_id = row[1]
                test_flag = int(row[3])

                # A simple resampling method to create more text-vision-touch triplets for GelSight sensor
                for tt in range(2):
                    if test_flag == 1:
                        self.text_list.append(-1)
                    else:
                        self.text_list.append(text_dir + 'tag_' + row[2] +'.pt')
                    self.vision_list.append(TAG_dir + folder + '/video_frame/' + image_id)
                    self.touch_list.append(TAG_dir + folder + '/gelsight_frame/' + image_id)

        for item in os.listdir(SSVTP_dir+'/images_tac/'):
            image_id = item.split('_')[1]
            tactile_path = SSVTP_dir+'/images_tac/'+item
            image_path = SSVTP_dir+'/images_rgb/'+item.replace('tac', 'rgb')
            self.text_list.append(text_dir + 'ssvtp_' + image_id +'.pt')
            self.touch_list.append(tactile_path)
            self.vision_list.append(image_path)



        # if args.train_num_samples is None:
        #     args.train_num_samples = len(os.listdir(os.path.join(self.data_root, "touch")))
        # print(args.train_num_samples)

        # for i in range(args.train_num_samples):
        #     self.touch_list.append(f"image_{i}_tac.jpg")
        #     self.vision_list.append(f"image_{i}_rgb.jpg")
            
        #     self.ids = self.id2title_folder_caps[:args.train_num_samples]
        # else:
        #     self.ids = self.id2title_folder_caps

        self.tokenizer = get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir)
        self.vision_transform = get_vision_transform(args)
        self.touch_transform = get_touch_transform(args)

    def __len__(self):
        return len(self.touch_list)

    def __getitem__(self, idx):
        try:
            sent_output, phra_output= self.get_text(idx)
            sent_input_ids, sent_attention_mask = sent_output
            phra_input_ids, phra_attention_mask = sent_output

            matched_modality_touch, matched_modality_vision = self.get_touch_vision(idx)
            return matched_modality_touch['pixel_values'], matched_modality_vision['pixel_values'], sent_input_ids, sent_attention_mask, phra_input_ids, phra_attention_mask        
        except Exception as error_msg:
            logging.info(f"Failed at {idx} with \"{error_msg}\"")
            return self.__getitem__(random.randint(0, self.__len__()-1))


    def get_text(self, id):
        if self.text_list[id] == -1:
            sent_output = (torch.zeros(77).int(), torch.zeros(77).int())
        else:
            sent_output = torch.load(self.text_list[id])
        phra_output = None
        return sent_output, phra_output
    
    def get_touch_vision(self, id):
        touch_path = self.touch_list[id]
        touch = load_and_transform_touch(touch_path, self.vision_transform)

        vision_path = self.vision_list[id]
        vision = load_and_transform_vision(vision_path, self.vision_transform)

        return touch,vision
    