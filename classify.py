#!/usr/bin/env python3

'''
A basic CNN classifier for butterfly classifications. We are using dataset: https://www.kaggle.com/veeralakrishna/butterfly-dataset as opposed to the much larger https://www.kaggle.com/c/fieldguide-challenge-moths-and-butterflies/leaderboard because the latter requires some cleaning and because my machine isn't quite powerful enough to process the huge amount of data in that set.
'''

import os

import numpy as np
import pandas as pd

import sklearn.preprocessing import LabelEncoder

categories = []
filenames = os.listdir('leedsbutterfly/images/')
for filename in filenames:
    category = filename.split(".")[0]
    categories.append(category[0:3])

cat = pd.DataFrame(categories)

la=LabelEncoder()
labels = la.fit_transform(cat[0])
