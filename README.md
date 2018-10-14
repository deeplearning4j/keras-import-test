# keras-samediff-test

Test suite for testing keras -> samediff model import.

## Installing dependencies


### install [pyjnius](https://github.com/kivy/pyjnius)


### Install pydl4j

```bash
git clone https://www.github.com/deeplerning4j/pydl4j.git
cd pydl4j
python setup.py install
```

### Install maven

```bash
apt-get install maven
```



### Build pydl4j jar

```bash
pydl4j install
```


### install jumpy

```
git clone https://www.github.com/deeplerning4j/jumpy.git
cd jumpy
python setup.py install
```

### install other python dependencies

```
pip install numpy
pip install tensorflow
pip install keras
```

## Usage

* Save keras models to `models` directory
* `python run.py`


## Continuous integration

For each keras model file that was succesfully converted to TF protobuff, a directory will be created in the `outputs` directory.
For example, for the test file 'my_model.h5', the exporteed files will be in 'outputs/my_model'.
It will contain:

* `my_model.pb`
* `my_model.pb.txt`
* `tests.json`
* Bunch of `.npy` files

`tests.json` contains a list of test cases. Each test case is a dictionary:`{inputs: [a.npy, b.npy,..], expected_outputs: [c.npy, d.npy,...]}`



