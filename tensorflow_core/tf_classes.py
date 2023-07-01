import tensorflow as tf

class Normalize(tf.Module):

    def __init__(self, x):
        self.mean = tf.math.reduce_mean(x, axis = 0)
        self.std = tf.math.reduce_std(x, axis = 0)

    def norm(self, x):
        return (x - self.mean)/self.std

    def unnorm(self, x):
        return (x * self.std) + self.mean

class LinearRegression(tf.Module):

    def __init__(self):
        self.built = False

    @tf.function
    def __call__(self, x):

        if not self.built:
            rand_w = tf.random.uniform(shape = [x.shape[-1], 1])
            rand_b = tf.random.uniform(shape = [])
            self.w = tf.Variable(rand_w)
            self.b = tf.Variable(rand_b)
            self.built = True

        # We are adding these tensors together
        y = tf.add(tf.matmul(x, self.w), self.b)
        return tf.squeeze(y, axis = 1)

class LogisticRegression(tf.Module):

    def __init__(self):
        self.built = False

    def __call__(self, x, train = True):

        if not self.built:
            rand_w = tf.random.uniform(shape = [x.shape[-1], 1], seed = 22)
            rand_b = tf.random.uniform(shape = [], seed = 22)
            self.w = tf.Variable(rand_w)
            self.b = tf.Variable(rand_b)

            self.built = True

        z = tf.add(tf.matmul(x, self.w), self.b)
        z = tf.squeeze(z, axis = 1)
        if train:
            return z
        return tf.sigmoid(z)

class ExportModule(tf.Module):
  def __init__(self, model, extract_features, norm_x, norm_y):
    # Initialize pre and postprocessing functions
    self.model = model
    self.extract_features = extract_features
    self.norm_x = norm_x
    self.norm_y = norm_y

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
  def __call__(self, x):
    # Run the ExportModule for new data points
    x = self.extract_features(x)
    x = self.norm_x.norm(x)
    y = self.model(x)
    y = self.norm_y.unnorm(y)
    return y

class Adam:

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.ep = ep

        self.t = 1.
        self.v_dvar, self.s_dvar = [], []
        self.built = False

    def apply_gradients(self, grads, vars):
        # Initializing these variables on the first call
        if not self.built:
            for var in vars:
                v = tf.Variable(tf.zeros(shape = var.shape))
                s = tf.Variable(tf.zeros(shape = var.shape))
                self.v_dvar.append(v)
                self.s_dvar.append(s)
            self.built = True

        # Update the model variables given the gradients
        for i, (d_var, var) in enumerate(zip(grads, vars)):
            self.v_dvar[i].assign(self.beta_1*self.v_dvar[i] + (1-self.beta_1)*d_var)
            self.s_dvar[i].assign(self.beta_2*self.s_dvar[i] + (1-self.beta_2)*tf.square(d_var))
            v_dvar_bc = self.v_dvar[i]/(1-(self.beta_1**self.t))
            s_dvar_bc = self.s_dvar[i]/(1-(self.beta_2**self.t))
            var.assign_sub(self.learning_rate*(v_dvar_bc/(tf.sqrt(s_dvar_bc) + self.ep)))

        self.t += 1.
        return
