#!/usr/bin/python3

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read and clean
df_merc = pd.read_table('./train.tsv')
df_merc = df_merc.drop(df_merc[df_merc.price == 0].index)
df_cats = df_merc.category_name.str.split('/')
cats = ['first_cat', 'second_cat', 'third_cat']

for i in range(3):
    df_merc[cats[i]] = df_cats.str.get(i)

