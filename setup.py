import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuralplayground",
    version="0.0.1",
    author="clem_domine_rod_carrasco",
    author_email="rodrigo.carrasco.davis@gmail.com",
    description="A Standardized Environment for Hippocampus and Entorhinal Cortex models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ClementineDomine/EHC_model_comparison",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)