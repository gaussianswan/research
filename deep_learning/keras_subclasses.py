import tensorflow as tf

from tensorflow import keras

class Linear(keras.layers.Layer):

    def __init__(self, units = 32):
        super().__init__()
        self.units = units

    # This function is going to be called in the __call__ method of the parent class
    def build(self, input_shape):

        self.w = self.add_weight(
            shape = (input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )

        self.b = self.add_weight(
            shape = (self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# We can create a unit which is a block of perceptrons
class MLPBlock(keras.layers.Layer):

    def __init__(self):
        super().__init__()

        self.linear_1 = Linear(32)
        self.linear_2 = Linear(10)
        self.linear_3 = Linear(5)

    def call(self, inputs):

        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        x = self.linear_3(x)
        return tf.nn.tanh(x)


class Sampling(keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape = (batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(keras.layers.Layer):
    """Maps MNIST digits to a triplet of (z_mean, z_log_var, z)

    Args:
        layers (_type_): _description_
    """
    def __init__(self, latent_dim = 32, intermediate_dim = 64, name = 'encoder', **kwargs):
        super().__init__(name = name, **kwargs)

        self.dense_proj = keras.layers.Dense(intermediate_dim, activation = 'relu')
        self.dense_mean = keras.layers.Dense(latent_dim)
        self.dense_log_var = keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(keras.layers.Layer):

    def __init__(self, original_dim, intermediate_dim = 64, name = 'decoder', **kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_proj = keras.layers.Dense(intermediate_dim, activation = 'relu')
        self.dense_output = keras.layers.Dense(original_dim, activation = 'sigmoid') # This is what we are outputting

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)

class VariationalAutoEncoder(keras.Model):

    """
    Combines the encoder and decoder into one end-to-end model
    """

    def __init__(self, original_dim, intermediate_dim = 64, latent_dim = 32, name = 'autoencoder', **kwargs):
        super().__init__(name = name, **kwargs)

        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed


if __name__ == "__main__":

    x = tf.ones((2, 2))
    linear_layer = Linear(units=3)
    print(linear_layer(x))

    mlp = MLPBlock()
    print(mlp(x))

    # Training the variational autoencoder here
    original_dim = 784
    vae = VariationalAutoEncoder(original_dim=original_dim, intermediate_dim=64, latent_dim=32)

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    mse_loss_fn = keras.losses.MeanSquaredError()
    loss_metric = keras.metrics.Mean()

    (x_train, _), _ = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255

    # We create a train dataset where we take the mnist data and batch it into 64
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    EPOCHS = 2
    for epoch in range(EPOCHS):

        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)

                loss = mse_loss_fn(x_batch_train, reconstructed)
                loss += sum(vae.losses)

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

    # At the end of this, we have an autoencoder which can take the image, condense it into some low level representation,
    # and then reconstruct the image.