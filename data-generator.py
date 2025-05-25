import pandas as pd
import numpy as np
from faker import Faker
from sklearn.datasets import make_classification
import sys

# Seed for reproducibility
np.random.seed(42)
faker = Faker()

# Dataset size and chunking
# 0: 1000000 rows + 1 for header
# 1: 30000000  + 1 for header
# 2: 60000000  + 1 for header
# 3: 100000000 + 1 for header

num_samples = int(sys.argv[1])  # Total rows
chunk_size = 1000000   # Process in chunks
header_written = False    # Flag to ensure header is only written once

# Process in chunks
for i in range(0, num_samples, chunk_size):
    # Generate numerical features
    X, y = make_classification(n_samples=chunk_size, 
                               n_features=3, 
                               n_informative=3, 
                               n_redundant=0, 
                               n_clusters_per_class=1, 
                               random_state=42)

    # Generate categorical features
    category_1 = (faker.color_name() for _ in range(chunk_size))
    category_2 = np.random.choice(['A', 'B', 'C', 'D'], chunk_size)

    df = pd.DataFrame({
        'num_feature_1': X[:, 0],
        'num_feature_2': X[:, 1],
        'num_feature_3': X[:, 2],
        'category_1': list(category_1),
        'category_2': category_2,
        'label': y
    })

    # Stream to stdout
    df.to_csv(sys.stdout, index=False, header=not header_written)
    header_written = True
