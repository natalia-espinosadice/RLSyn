
import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self, H):
        super().__init__() 
        self.out_dim = H.DIMENSION
        self.cond_dim = H.CONDITIONS_DIMENSION
        self.hidden_dim = H.G_H
        self.n_layers = H.G_HLAYERS

        self.hidden = [tf.keras.layers.Dense(self.hidden_dim, activation = None) for _ in range (self.n_layers)]
        self.norms = [tf.keras.layers.BatchNormalization(epsilon=1e-5) for _ in range (self.n_layers)]
        self.out = tf.keras.layers.Dense(self.out_dim, activation=tf.nn.sigmoid)

    def call(self, z, training=False):
        x = z                          
        for dense, bn in zip(self.hidden, self.norms):
            x = tf.nn.relu(bn(dense(x), training=training))
        return self.out(x)
        
    def test(self, z): 
        y = self.call(z, training=False)
        if self.cond_dim > 0: 
            cond = tf.cast(y[:, :self.cond_dim] > 0.5, tf.float32)
            rest = y[:, self.cond_dim:]
            y = tf.concat([cond, rest], axis=-1)
        return y
 
class Discriminator(tf.keras.Model):
    def __init__(self, H):
        super().__init__()
        self.hidden_dim = H.D_H
        self.n_layers = H.D_HLAYERS

        self.hidden = [tf.keras.layers.Dense(self.hidden_dim, activation=None) for _ in range(self.n_layers)]
        self.norms = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(self.n_layers)]
        self.out   = tf.keras.layers.Dense(1) 
    
    def call(self, x): 
        for dense, ln in zip(self.hidden, self.norms): 
            x = tf.nn.relu(ln(dense(x)))
        return self.out(x) 