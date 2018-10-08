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

## install other python dependencies

```
pip install numpy
pip install tensorflow
pip install keras
```

## Usage

* Save keras models to `models` directory
* `python run.py`
