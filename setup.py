from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vietnamese_address_classification",
    version="0.1",
    author="XDcobra",
    author_email="your.email@example.com",
    description="A package for parsing and classifying Vietnamese addresses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XDcobra/vietnamese_address_classification",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "rapidfuzz>=3.0.0",
    ],
) 