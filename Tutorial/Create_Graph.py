import tensorflow as tf
c = tf.constant(4.0)
assert c.graph is tf.get_default_graph()