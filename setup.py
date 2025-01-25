import os

from setuptools import setup, find_packages

base_packages = ["numpy", "pandas", "pymc"]
plot_packages = ["matplotlib"]
dev_packages = ["pytest", "hypothesis"]


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='usopp',
    packages=find_packages(where='src'),
    package_dir={"": "usopp"},
    install_requires=base_packages,
    extras_require={
      "dev": dev_packages,
      "plot": plot_packages
    },
    description='An hierarchical version of Facebooks prophet in PyMC3',
    author='Matthijs Brouns',
    long_description=read('readme.md'),
    long_description_content_type='text/markdown',
)
