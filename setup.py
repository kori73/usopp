import os

from setuptools import setup, find_packages

base_packages = ["numpy", "pandas", "pymc"]
plot_packages = ["matplotlib"]
dev_packages = ["pytest", "hypothesis", "nbconvert", "jupyter", "ipykernel"]
docs_packages = [
    "sphinx",
    "myst-nb",
    "sphinx-autodoc-typehints",
]

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="usopp",
    packages=find_packages(where="."),
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages,
        "plot": plot_packages,
        "docs": docs_packages,
    },
    description="A hierarchical version of Facebook's Prophet in PyMC3",
    author="Koray Beyaz",
    long_description=read("readme.md"),
    long_description_content_type="text/markdown",
)
