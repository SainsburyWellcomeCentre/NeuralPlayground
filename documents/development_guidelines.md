# Development Guidelines

## Formatting and Pre-commit Hooks

Running `pre-commit install` will set up [pre-commit hooks](https://pre-commit.com/) to ensure a consistent formatting style. Currently, these include:
* [ruff](https://github.com/charliermarsh/ruff) does a number of jobs, including enforcing PEP8 and sorting imports
* [black](https://black.readthedocs.io/en/stable/) for auto-formatting
* [mypy](https://mypy.readthedocs.io/en/stable/index.html) as a static type checker
* [check-manifest](https://github.com/mgedmin/check-manifest) to ensure that the right files are included in the pip package.

These will prevent code from being committed if any of these hooks fail. To run them individually (from the root of the repository), you can use:

    ruff .
    black ./
    mypy -p movement
    check-manifest
To run all the hooks before committing:

    pre-commit run  # for staged files
    pre-commit run -a  # for all files in the repository

For docstrings, we adhere to the numpydoc style.

## Testing
We use [pytest](https://docs.pytest.org/en/latest/) for testing and aim for ~100% test coverage (as far as is reasonable). All new features should be tested. Write your test methods and classes in the tests folder.

For some tests, you will need to use real experimental data. Do not include these data in the repository, especially if they are large. We store several sample datasets in an external data repository. See [sample data](https://neuroinformatics-unit.github.io/movement/contributing.html#sample-data) for more information.

## Continuous Integration
All pushes and pull requests will be built by [GitHub actions](https://docs.github.com/en/actions). This will usually include linting, testing and deployment.

A GitHub actions workflow (`.github/workflows/test_and_deploy.yml`) has been set up to run (on each push/PR):
* Linting checks (pre-commit).
* Testing (only if linting checks pass)
* Release to PyPI (only if a git tag is present and if tests pass).

## Versioning and Releases
We use [semantic versioning](https://semver.org/), which includes `MAJOR`.`MINOR`.`PATCH` version numbers:
* PATCH = small bugfix
* MINOR = new feature
* MAJOR = breaking change

We use [setuptools_scm](https://github.com/pypa/setuptools_scm) to automatically version movement. It has been pre-configured in the `pyproject.toml` file. `setuptools_scm` will automatically [infer the version using git](https://github.com/pypa/setuptools_scm#default-versioning-scheme). To manually set a new semantic version, create a tag and make sure the tag is pushed to GitHub. Make sure you commit any changes you wish to be included in this version. E.g. to bump the version to `1.0.0`:

    git add .
    git commit -m "Add new changes"
    git tag -a v1.0.0 -m "Bump to version 1.0.0"
    git push --follow-tags

Alternatively, you can also use the GitHub web interface to create a new release and tag.

The addition of a GitHub tag triggers the package’s deployment to PyPI. The version number is automatically determined from the latest tag on the main branch.
