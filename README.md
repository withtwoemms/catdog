# catdog
a suite for training models that know cats from dogs


# Setup

### Using virtual environments

[pyenv](https://github.com/pyenv/pyenv) is a great tool for creating self-contained python environments.
Rather than sully your system python's dependency namespace, use a virtual environment.
[Install pyenv](https://github.com/pyenv/pyenv#installation).
Once installed, wiew available python versions to install:

```
pyenv install -l
```

and then install one (preferably a recent version--i.e. >3.6):

```
pyenv install 3.10.7
```

Now to create the virtual environment.
The following will create a virtual environment called "catdog":

```
pyenv virtualenv 3.10.7 catdog
```

The created environment can be activated with:

```
pyenv activate catdog
```

Now that you've got an active virtual environment, you can install this project and it's console scripts:

```
pip install .
```

This will package the current project's current working state and install the `catdog` project along with its dependencies.
If you're wanting to avoid the need to re-install after changing project files, please do install using `pip`'s ["editable mode"](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).

# Usage

`catdog` comes which a couple of [console scripts](https://python-packaging.readthedocs.io/en/latest/command-line-scripts.html):

* `predict`
* `train`

Future work will improve the ergonomics of these scripts.
The get the packaged model (clerverly named "model.h5") to classify your sample image, place an image of a cat or dog in the "data/samples/" directory and run something like this:

```
predict model.h5 <sample-file-name>
```
