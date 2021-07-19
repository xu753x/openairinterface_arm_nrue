# How to contribute in the OAI new python framework. #

The [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/) is authoritative.

## Documentation ##

*  Document all public functions and *keep those docs up to date* when you make changes.
*  We use Google style docstrings in our codebase

Example:

```python
def foo(arg1: str) -> int:
    """Returns the length of arg1.

    Args:
        arg1 (str): string to calculate the length of

    Returns: the length of the provided parameter
    """
    return len(arg1)
```

## Formatters ##

* We recommend to use `autopep8`, `isort` and `add-trailing-comma` as they conform to pep8.
* We do not recommend other formatters such as `black`, as it diverges from pep8 on basic things like line length, etc.

### Installation of `isort` ###

See for details at [isort](https://pypi.org/project/isort/)

```bash
pip install isort
```

Usage:

```bash
isort oai-ci-test-main.py
```

### Installation of `autopep8` ###

See for details at [autopep8](https://pypi.org/project/autopep8/)

```bash
pip install --upgrade autopep8
```

Usage:

```bash
autopep8 --select W191,W291,W292,W293,W391,E2,E3 -r --in-place oai-ci-test-main.py
```

### Installation of `add-trailing-comma` ###

See for details at [add-trailing-comma](https://pypi.org/project/add-trailing-comma/)

```bash
pip install add-trailing-comma
```

Usage:

```bash
add-trailing-comma --py35-plus --exit-zero-even-if-changed oai-ci-test-main.py
```

### Linter ###

Formatting does not mean your code is pep8-compliant.

See for details at [flake8](https://pypi.org/project/flake8/)

```bash
pip install flake8
```

Usage:

You shall be in this folder so it can use `setup.cfg` file as configuration file.

```bash
cd ci-scripts/python_v2
flake8 oai-ci-test-main.py
```

You shall have no error message.
