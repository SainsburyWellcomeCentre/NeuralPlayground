# Development Guidelines

## Formatting and Pre-commit Hooks

Running `pre-commit install` will set up [pre-commit hooks](https://pre-commit.com/) to ensure a consistent formatting style. Currently, these include:

* [ruff](https://github.com/astral-sh/ruff) does a number of jobs, including code linting and auto-formatting.
* [check-manifest](https://github.com/mgedmin/check-manifest) to ensure that the right files are included in the pip package.

These will prevent code from being committed if any of these hooks fail.

To run all the hooks before committing:

```bash
pre-commit run  # for staged files
pre-commit run -a  # for all files in the repository
```

Some problems will be automatically fixed by the hooks. In this case, you should stage the auto-fixed changes and run the hooks again:

```bash
git add .
pre-commit run
```

If a problem cannot be auto-fixed, the corresponding tool will provide information on what the issue is and how to fix it.

For docstrings, we adhere to the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style. Make sure to provide docstrings for all public functions, classes, and methods.

## Testing
We use [pytest](https://docs.pytest.org/en/latest/) for testing and aim for ~100% test coverage (as far as is reasonable). All new features should be tested. Write your test methods and classes in the tests folder.

You can run the tests locally by executing `pytest` in the repository root, while in an active [development environment](../README.md#install-for-development).

For some tests, you will need to use real experimental data. Do not include these data in the repository, especially if they are large. We store several sample datasets in an [external data repository](https://gin.g-node.org/SainsburyWellcomeCentre/NeuralPlayground).

## Continuous Integration
All pushes and pull requests will be built by [GitHub actions](https://docs.github.com/en/actions). This will usually include linting, testing and deployment.

A GitHub actions workflow (`.github/workflows/test_and_deploy.yml`) has been set up to run (on each push/PR):
* Linting checks (pre-commit).
* Testing (only if linting checks pass)
* Release to PyPI (only if a git tag is present and if tests pass).

## Versioning
We use [semantic versioning](https://semver.org/), which includes `MAJOR`.`MINOR`.`PATCH` version numbers:
* PATCH = small bugfix
* MINOR = new feature
* MAJOR = breaking change

> [!note]
> Currently, we only apply [semantic versioning](https://semver.org/) to MINOR and PATCH versions. Until we reach our first stable `v1` release we leave the MAJOR version as 0, even for breaking changes.

We use [setuptools_scm](https://github.com/pypa/setuptools_scm) to automatically version NeuralPlayground. It has been pre-configured in the `pyproject.toml` file. `setuptools_scm` will automatically [infer the version using git](https://github.com/pypa/setuptools_scm#default-versioning-scheme). 


## Releasing a new version

The addition of a GitHub tag triggers the package's deployment to PyPI.
The version number is automatically determined from the latest tag on the main branch.
Below you will find the **steps to create a new release** using GitHub's release interface.

1. Start a draft release
    - Go to <https://github.com/SainsburyWellcomeCentre/NeuralPlayground/releases/new>, which is equivalent to going on GitHub [Releases](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/releases) and clicking **Draft a new release**.

        
2. Set the tag and title
    - In the **Tag** dropdown, create a new tag following [semantic versioning](https://semver.org/). Don't forget to include the `v` prefix, e.g. `v0.3.0`.
    - Leave **Target** set to `main`.
    - Use the tag name as the release title, e.g. `v0.3.0`.
        
3. Generate release notes
    - Leave _previous tag_ set to **Auto** and click **Generate release notes**.
    - Optionally, organise the PR list under **What's Changed** into meaningful subsections (`###` headers).
    
    > [!tip]
    >    - Acknowledge first-time or external contributors.
    >    - For breaking changes, include code snippets showing old vs new syntax.

4. Publish the release
    - Leave _Set as the latest release_ checked (default).
    - After publishing, check the **Actions** tab to ensure both workflows finish successfully.

5. Verify on PyPI
    - Confirm the new version appears on [PyPI](https://pypi.org/project/NeuralPlayground/).

6. Optional: announce the release on social media.

