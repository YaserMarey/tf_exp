import tensorflow.keras as k

# 1- Initialize the model, one layer fully connected, with one node
model = k.Sequential(k.layers.Dense(units=1, input_shape=([1])))

# 2- Compile the model with stochastic gradient descent optimizer to fin minimum of loss
# function of mean squared error
model.compile(optimizer='sgd', loss='mean_squared_error')


# 3- Input is 6 values vector x, and labels is vector y
# relationship is y=3*x + 1
x = [1.0, 0.0, 2.0, 3.0, 6.0, -4.0]
y = [4.0, 0.0, 7.0, 10.0, 19.0, -11.0]

# 4- Train the model, pass through the entire dataset of x, y pairs for 500 times
model.fit(x, y, epochs=500)

# 5- Check with a new value not seen before, expected output is 31.0
print(model.predict([10.0]))


