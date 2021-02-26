# Poems from: https://www.kaggle.com/johnhallman/complete-poetryfoundationorg-dataset


import os

import pandas as pd
import numpy as np



df = pd.read_csv("kaggle_poem_dataset.csv")

sample_num = 10

idcs = np.random.randint(0, len(df), sample_num)

sample_poems = df.iloc[idcs]["Content"]

name_prefix = f"poems_{sample_num}_"

os.makedirs("poems", exist_ok=True)

previous_samples = os.listdir("poems")
previous_samples = [f[len(name_prefix):-len(".txt")] for f in previous_samples if f.startswith(name_prefix) and f.endswith(".txt")]
previous_nums = [int(f) for f in previous_samples]
max_num = max(previous_nums) if len(previous_nums) > 0 else -1

current_num = max_num + 1

name = f"{name_prefix}{current_num}.txt"
path = os.path.join("poems", name)


out_string = ""
for poem in list(sample_poems):
    poem = poem.replace("\n", " ")
    out_string += poem + "\n"

with open(path, "w+") as f:
    f.write(out_string)
#sample_poems.to_csv(path)
