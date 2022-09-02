from setuptools import setup, find_packages

setup(
    name="rofunc",
    version="0.1",
    author="skylark",
    author_email="jjliu@mae.cuhk.edu.hk",
    requires= ['numpy','matplotlib', 'mvnx'],
    packages=find_packages(),
    license="apache 3.0"
)