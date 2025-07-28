from scipy.io import arff
import pandas as pd

# Load the .arff file
data, meta = arff.loadarff('./compas.arff')

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('./compas_prepared_data.csv', index=False)
