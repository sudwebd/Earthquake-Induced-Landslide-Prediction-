import pandas as pd
import random

# The data to load
f = "/home/btpbatch3/Desktop/BTP3/April/Rough work/add_neg_toh.csv"

# Count the lines
num_lines = sum(1 for l in open(f))

# Sample size - in this case ~10%
size = 12000

# The row indices to skip - make sure 0 is not included to keep the header!
skip_idx = random.sample(range(1, num_lines), num_lines - size)

# Read the data
data = pd.read_csv(f, skiprows=skip_idx)

data.to_csv("0.csv", encoding='utf-8', index = False)