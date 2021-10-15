import pandas as pd
import urllib.request
import numpy as np
from PIL import Image
import os
import shutil

datasets = ['amazon_boys_girls', 'amazon_men', 'pinterest']
whiteFrame = 255 * np.ones((224, 224, 3), np.uint8)
whiteImage = Image.fromarray(whiteFrame)

for dataset in datasets:
    if os.path.exists(f'{dataset}/original/images/'):
        shutil.rmtree(f'{dataset}/original/images/')
    os.makedirs(f'{dataset}/original/images/')

    all_items = pd.read_csv(f'{dataset}/all_items.csv')
    items = pd.read_csv(f'{dataset}/items.tsv', sep='\t', header=None)
    for index, row in all_items.iterrows():
        item_id = items[items[0] == row['ASIN']].iloc[0][1]

        try:
            urllib.request.urlretrieve(row['URL'], f'{dataset}/original/images/{item_id}.jpg')
        except Exception as ex:
            print(f'Image not Available. Saving white image for {item_id}')
            whiteImage.save(f'{dataset}/original/images/{item_id}.jpg')

        if (index + 1) % 10 == 0:
            print(f'{index + 1}/{len(items)}')
