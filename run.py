from keras2tf import keras_to_tf
from numpy.testing import assert_allclose
import os
import jumpy as jp
import numpy as np
import tensorflow as tf
import click
import time
import keras
import shutil
from pb2txt import pb2txt
import json


jp.set_context_dtype('float32')


current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
output_dir =  os.path.join(current_dir, 'output')

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


def _replace_none(shape):
    return [x if x is not None else 32 for x in shape]


def _replace_special_chars(x):
    y = ''
    for c in x:
        if not ((c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z') or (c >= '0' and c <= '9')):
            y += '_'
        else:
            y += c
    return y


def get_test_data(keras_model):
    input_shapes = keras_model.input_shape
    if not isinstance(input_shapes, list):
        input_shapes = [input_shapes]
    input_shapes = [_replace_none(s) for s in input_shapes]
    inputs = []

    X = []
    for input_shape in input_shapes:
        X.append(np.zeros(input_shape, 'float32'))
    inputs.append(X)

    X = []
    for input_shape in input_shapes:
        X.append(np.ones(input_shape, 'float32'))
    inputs.append(X)

    X = []
    for input_shape in input_shapes:
        X.append(np.cast['float32'](np.random.random(input_shape)))
    inputs.append(X)

    expected_outputs = []
    for x in inputs:
        y_keras = keras_model.predict(x)
        if not isinstance(y_keras, list):
            y_keras = [y_keras]
        expected_outputs.append(y_keras)
    return inputs, expected_outputs


def numeric_test_sd(model, inputs, expected_outputs):
    for inps, exp_outs in zip(inputs, expected_outputs):
        outs = model(inps)
        if not isinstance(outs, list):
            outs = [outs]
        for out, exp_out in zip(outs, exp_outs):
            out= out.numpy()
            assert_allclose(out, exp_out, 1e-3)


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

if len(models_to_test) == 0:
    click.secho("No test files found!", fg='red')
    exit()

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
            tf_pb, input_names, output_names = keras_to_tf(model)
        except Exception as e:
            num_failed += 1
            overall_report.append(click.style(model, fg='red'))
            error = [model, "Failed during converting keras model to tensorflow protobuff.", e]
            errors.append(error)
            continue
        try:
            imported_model = jp.TFModel(tf_pb, input_names, output_names)
        except Exception as e:
            num_failed += 1
            overall_report.append(click.style(model, fg='red'))
            error = [model, "Failed during import tensorflow model to SameDiff.", e]
            errors.append(error)
            continue
        sess = tf.Session()
        keras.backend.set_session(sess)
        keras_model = keras.models.load_model(model)
        inputs, expected_outputs = get_test_data(keras_model)
        test_name = os.path.basename(model)[:-3]
        test_dir = os.path.join(output_dir, test_name)
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)
        os.mkdir(test_dir)
        pb_txt = os.path.join(test_dir, test_name + '.pb.txt')
        pb = os.path.join(test_dir, test_name + '.pb')
        pb2txt(tf_pb, pb_txt)
        shutil.copyfile(tf_pb, pb)

        tests = []
        for i, (inp_arrs, out_arrs) in enumerate(zip(inputs, expected_outputs)):
            test = {}
            test['inputs'] = {}
            test['expected_outputs'] = {}
            for inp_name, inp_arr in zip(input_names, inp_arrs):
                arr_file = 'arr_inp_' + _replace_special_chars(inp_name) + '_' + str(i) + '.npy'
                arr_path = os.path.join(test_dir, arr_file)
                np.save(arr_path, inp_arr)
                test['inputs'][inp_name] = arr_file
            for out_name, out_arr in zip(output_names, out_arrs):
                arr_file = 'arr_out_' + _replace_special_chars(out_name) + '_' + str(i) + '.npy'
                arr_path = os.path.join(test_dir, arr_file)
                np.save(arr_path, out_arr)
                test['expected_outputs'][out_name] = arr_file
            tests.append(test)
        tests_json = os.path.join(test_dir, 'tests.json')
        with open(tests_json, 'w') as f:
            json.dump(tests, f)
        try:
            numeric_test_sd(imported_model, inputs, expected_outputs)
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