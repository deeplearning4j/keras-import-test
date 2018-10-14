import tensorflow as tf
import os

def pb2txt(pb_file, txt_file=None):
    if txt_file is None:
        txt_file = pb_file + '.txt'
    with tf.gfile.GFile(pb_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.train.write_graph(graph_def,os.path.dirname(txt_file), txt_file, True)
    return txt_file
