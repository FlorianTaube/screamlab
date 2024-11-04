import os
import setuptools


def read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as file:
        content = file.read()
    return content


setuptools.setup(
    name="CorziliusNMR",
    version=read("VERSION").strip(),
    description="",
    long_description=read("README.rst"),
    long_description_content_type="text/x_data-rst",
    author="Florian Taube",
    author_email="florian.taube@uni-rostock.de",
    url="",
    project_urls={
        "Documentation": "",
        "Source": "",
    },
    packages=setuptools.find_packages(exclude=("tests", "docs")),
    license="BSD",
    keywords=[
        "Buildup Time Calculator",
        "NMR",
        "DNP",
        "SCREAM_DNP",
        "TopspinExport",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
    ],
    install_requires=[
        "matplotlib",
        "numpy",
        "lmfit",
        "scipy",
    ],
    extras_require={
        "dev": [
            "prospector",
            "pyroma",
            "bandit",
            "black",
            "pymetacode",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "sphinx_multiversion",
        ],
        "deployment": [
            "build",
            "twine",
        ],
    },
    python_requires=">=3.7",
    include_package_data=True,
)
