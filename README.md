# cell-tracker

adaptation of bnoi/scikit-tracker to track cells

## Depencies

### Required

* numpy >= 1.8
* scipy >= 0.12
* pandas >= 0.13
* scikit-image >= 0.9
* scikit-learn >= 0.13
* matplotlib >= 1.8
* sktracker >= 0.2-dev


### Optionnal (for UI elements)

* ipython >= 2.0
* pyqt4 >= 4.10

# Travis
[![Build Status](https://travis-ci.org/bnoi/cell-tracker.png?branch=master)](https://travis-ci.org/bnoi/cell-tracker)


# Install

First, install the requirements (except `sktracker`) from pip or anaconda.

The simplest way is to use [anaconda](https://store.continuum.io/cshop/anaconda/),
Once you installed anaconda, create a python 3 environment:

```bash
$ conda create -n py33 python=3.3 anaconda
```

And activate it:

```bash
$ source activate py33
```

Your bash prompt should now start with `(py33)`
You'll need to install pyqt4 (make sure you're still within the py33 virtuall environment)

```bash
$ conda install pyqt4
```

The version of IPython is too hold for the UI tools in the notebooks,
so if you want those, install the latest ipython with pip:
```bash
$ pip install --upgrade ipython
```

Now create a `src` directory where you'll clone `scikit-tracker` and `cell-tracker`:

```bash
$ mkdir src
$ cd src
$ git clone https://bnoi/scikit-tracker.git
$ cd scikit-tracker
$ python setup.py install
$ cd ..
$ git clone https://bnoi/cell-tracker.git
$ cd cell-tracker
$ python setup.py install
```

If it complains about gcc, try installing build-essentials (if you're
using a debian-like distro).

# Usage

In the `cell-tracker` source folder, you'll find a `notebooks` folder.

Copy it all somewhere in your home folder, that's where you'll use the code.

Start an ipython notebook (still under the py33 virtual environment):
```
$ ipython notebook
```

Navigate to the notebooks folder you just copied, and start with the `Segmentation and Tracking` notebook.


