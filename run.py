from keras2tf import keras_to_tf
from numpy.testing import assert_allclose
import os
import jumpy as jp
import numpy as np
import tensorflow as tf
import click
import time
import keras



jp.set_context_dtype('float32')


current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')



def numeric_test(keras_model, imported_model):
    input_shape = keras_model.input_shape
    if isinstance(input_shape, list):
        raise Exception('Multi input models are not currently supported')
    input_shape = list(input_shape)
    for i, d in enumerate(input_shape):
        if d is None:
            input_shape[i] = 32
    inputs = []
    inputs.append(np.zeros(input_shape, 'float32'))
    inputs.append(np.ones(input_shape, 'float32'))
    inputs.append(np.cast['float32'](np.random.random(input_shape)))
    for x in inputs:
        y_keras = keras_model.predict(x)
        y_sd = imported_model(x).numpy()
        assert_allclose(y_sd, y_keras, 1e-5)


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
overall_report = []
num_passed = 0
num_failed = 0
start_time = time.time()

errors = []
with click.progressbar(models_to_test) as models:
    for model in models:
        try:
            tf_pb = keras_to_tf(model)
        except Exception as e:
            num_failed += 1
            overall_report.append(click.style(model, fg='red'))
            error = [model, "Failed during converting keras model to tensorflow protobuff.", e]
            errors.append(error)
            continue
        try:
            imported_model = jp.TFModel(tf_pb)
        except Exception as e:
            num_failed += 1
            overall_report.append(click.style(model, fg='red'))
            error = [model, "Failed during import tensorflow model to SameDiff.", e]
            errors.append(error)
            continue
        try:
            sess = tf.Session()
            keras.backend.set_session(sess)
            keras_model = keras.models.load_model(model)
            numeric_test(keras_model, imported_model)
        except Exception as e:
            num_failed += 1
            overall_report.append(click.style(model, fg='red'))
            error = [model, "Failed during numeric testing.", e]
            errors.append(error)
            continue
        num_passed += 1
        overall_report.append(click.style(model, fg='green'))
end_time = time.time()
time_taken = end_time - start_time
if errors:
    color = 'red'
    sm = ':('
else:
    color = 'green'
    sm = ':)'


overall_report.insert(0, click.style("{} tests run in {} seconds - {} passed, {} failed {}".format(len(models_to_test), 
                                                                                                time_taken, num_passed, num_failed, sm), fg=color))
overall_report.insert(1, click.style("\n", fg=color))
print('')                                                                                 
for x in overall_report:
    click.echo(x)


if errors:
    print('\n')
    click.secho("========= Errors =========", fg='red')
    for err in errors:
        print('')
        click.secho(err[0], fg='red')
        print('')
        click.secho(err[1], fg='red')
        click.secho(str(err[2]), fg='red')