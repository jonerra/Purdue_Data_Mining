from som import SOM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the input and target data sets
inputs = pd.read_csv('DataminingContest2009.Task2Inputs.Train.csv')
target = pd.read_csv('DataminingContest2009.Task2Targets.Train.csv')

# Concatenating the zip and state columns to reduce model bias
inputs["zipstate"] = inputs["zip1"].map(str) + inputs["state1"]

# Inputting the state missing value with the most common state
inputs['zipstate'] = inputs['zipstate'].fillna(inputs['zipstate'].mode()[0])

# Subset data to the most useful variables
inputs = inputs[['field3', 'flag5', 'zipstate', 'field1', 'field4']]
X = np.array(inputs)

# Train a 20x30 SOM with 400 iterations
som = SOM(4, 2, 3, 400)
som.train(inputs)

# Get output grid
image_grid = som.get_centroids()

# Map colours to their closest neurons
mapped = som.map_vects(X)

# Plot
plt.imshow(image_grid)
plt.title('Color SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], inputs.columns[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()
