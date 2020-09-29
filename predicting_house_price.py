# If house pricing was 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.
# Create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k

import tensorflow.keras as k
import numpy as np

def house_mode(y_new):
    # Training data
    xs = np.array([1,2,3,4,5], dtype=int)
    # Labels in hundreds of thousands of dollars
    ys = np.array([1.0,1.5,2.0,2.5,3.0], dtype=float)
    # The model that need to be learned is simple enough such that one node is sufficient
    model = k.Sequential([k.layers.Dense(units=1, input_shape=([1]))])
    # Use stochastic gradient descent and mean squared error
    model.compile(optimizer='sgd', loss='mean_squared_error')
    # Train
    model.fit(xs, ys, epochs=500)

    return model.predict(y_new)

no_of_rooms = 7
predicted_price = house_mode([no_of_rooms])
print('house of {0} rooms in hunderds of thousands of dolars is expected to be{1})'.
      format(no_of_rooms, predicted_price))