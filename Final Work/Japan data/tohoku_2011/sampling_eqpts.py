import pandas as pd
import random

# The data to load
f = "tohoku_0.csv"

# Count the lines
num_lines = sum(1 for l in open(f))

# Sample size - in this case ~10%
size = 3500

# The row indices to skip - make sure 0 is not included to keep the header!
skip_idx = random.sample(range(1, num_lines), num_lines - size)

# Read the data
data = pd.read_csv(f, skiprows=skip_idx)

data.to_csv("tohoku_eq0.csv", encoding='utf-8', index = False)