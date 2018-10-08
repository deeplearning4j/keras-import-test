# converts keras model h5 to TF protobuff


import keras
import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph
import keras.backend as K


assert keras.backend.backend() == 'tensorflow'


def keras_to_tf(keras_path, tf_path=None):
    if tf_path is None:
        tf_path = keras_path[:-2] + 'pb.txt'
    assert isinstance(keras_path, str)
    assert os.path.isfile(keras_path)
    temp_dir = "graph"
    checkpoint_prefix = os.path.join(temp_dir, "saved_checkpoint")
    checkpoint_state_name = "checkpoint_state"
    with tf.Session() as sess:
        keras.backend.set_session(sess)
        model = keras.models.load_model(keras_path)
        output_node_name = model.output.name.split(':')[0]
        saver = tf.train.Saver()
        checkpoint_path = saver.save(K.get_session(), checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)
        tf.train.write_graph(K.get_session().graph, os.path.dirname(tf_path), tf_path)
        freeze_graph.freeze_graph(
            tf_path,
            "",
            False,
            checkpoint_path,
            output_node_name,
            "save/restore_all",
            "save/Const:0",
            tf_path,
            False,
            "")
        '''
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_node_name])
        with tf.gfile.GFile(tf_path, "w") as f:
            f.write(output_graph_def.SerializeToString())
        '''
    return tf_path
