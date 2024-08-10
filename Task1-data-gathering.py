from datasets import load_dataset
import pandas as pd

# Stream data instead of downloading everything
dataset = load_dataset('HuggingFaceM4/WebSight', "v0.2", split='train', streaming=True)

# Define the number of examples to take
num_examples = 1000
data = []

# Stream data and collect examples
for i, example in enumerate(dataset):
    if i >= num_examples:
        break
    data.append(example)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('websight_data.csv', index=False)

