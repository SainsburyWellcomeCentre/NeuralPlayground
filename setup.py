import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuralplayground",
    version="0.6.0a1",
    author="Clementine Domine and Rodrigo Carrasco-Davis",
    author_email="rodrigo.carrasco.davis@gmail.com",
    licence="MIT",
    description="NeuralPlayground: A Standardised Environment for Evaluating Models of Hippocampus and Entorhinal Cortex",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ClementineDomine/NeuralPlayground",
    packages=setuptools.find_packages(),
    classifiers=(
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
