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
    install_requires=['nibabel','nipype','numpy', 'pandas', 'matplotlib', 'seaborn'],
    packages=find_packages(exclude=['cosanlab_preproc/tests']),
    license='MIT',
    # download_url='https://github.com/ljchang/emotionCF/archive/%s.tar.gz' %
    # __version__,
    **extra_setuptools_args
)