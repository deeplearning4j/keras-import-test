import os
import jumpy
from keras2tf import keras_to_tf
import click

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')


def is_h5_file(file):
    if file[-3:].lower() == '.h5':
        if os.path.isfile(file):
            return True
    return False


models_to_test = []

for f in os.listdir(models_dir):
    path = os.path.join(models_dir, f)
    if is_h5_file(path):
        models_to_test.append(path)


click.echo("Collected {} keras models to test:".format(len(models_to_test)))
for path in models_to_test:
    click.echo(path)


click.echo("Running tests...")

with click.progressbar(models_to_test) as models:
    for model in models:
        tf_pb = keras_to_tf(model)
        tfmodel = jumpy.TFModel(tf_pb)
