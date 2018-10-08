# converts keras model h5 to TF protobuff


import keras
import tensorflow as tf
import os


assert keras.backend.backend() == 'tensorflow'


def keras_to_tf(keras_path, tf_path=None):
    if tf_path is None:
        tf_path = keras_path[:-2] + 'pb.txt'
    assert isinstance(keras_path, str)
    assert os.path.isfile(keras_path)
    with tf.Session() as sess:
        keras.backend.set_session(sess)
        model = keras.models.load_model(keras_path)
        sess.run(tf.global_variables_initializer())
        output_node_name = model.output.name.split(':')[0]
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_node_name])
        with tf.gfile.GFile(tf_path, "w") as f:
            f.write(output_graph_def.SerializeToString())
    return tf_path
