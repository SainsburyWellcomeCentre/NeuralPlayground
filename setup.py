import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuralplayground",
    version="0.6b.0",
    author="Clementine Domine and Rodrigo Carrasco-Davis",
    author_email="rodrigo.carrasco.davis@gmail.com",
    description="NeuralPlayground: A Standardised Environment for Evaluating Models of Hippocampus and Entorhinal Cortex",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ClementineDomine/NeuralPlayground",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)