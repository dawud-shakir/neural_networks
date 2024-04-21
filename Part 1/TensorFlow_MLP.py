# MLP.py

import tensorflow as tf

# Define the MLP model
class MLP(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x

# Example usage
input_size = 10
hidden_size = 20
output_size = 1

model = MLP(input_size, hidden_size, output_size)

# Define loss function and optimizer
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Example data
inputs = tf.random.normal((5, input_size))  # Batch size of 5
targets = tf.random.normal((5, output_size))

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.numpy():.4f}')
```
