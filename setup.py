from cosanlab_preproc.version import __version__
from setuptools import setup, find_packages

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
    install_requires=['nibabel','nipype','numpy', 'pandas', 'nltools','matplotlib', 'seaborn','nipype','pybids'],
    packages=find_packages(exclude=['cosanlab_preproc/tests']),
    package_data={'cosanlab_preproc':['resources/*']},
    license='MIT',
    **extra_setuptools_args
)
