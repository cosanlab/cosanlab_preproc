from cosanlab_preproc.version import __version__
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name="cosanlab_preproc",
    version=__version__,
    description="Preprocessing tools for cosanlab",
    maintainer='Luke Chang',
    maintainer_email='luke.j.chang@dartmouth.edu',
    url='http://github.com/cosanlab/cosanlab_preproc',
    install_requires=requirements,
    packages=find_packages(exclude=['cosanlab_preproc/tests']),
    package_data={'cosanlab_preproc':['resources/*']},
    license='MIT',
    **extra_setuptools_args
)
