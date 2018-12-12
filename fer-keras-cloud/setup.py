#__author__ = ritvikareddy2
#__date__ = 2018-12-11

from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    # 'opencv-python',
    # 'cmake',
    # 'dlib',
    'matplotlib',
    'keras',
    'h5py',
    # 'scikit-image'
]

setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    description='run keras on gcloud ml-engine',
    author='Ritvika Nagula',
    author_email='ritvikareddy18@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    zip_safe=False)