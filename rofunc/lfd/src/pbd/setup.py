from setuptools import setup, find_packages
import sys

if sys.version_info[0] == 2:
    # Setup for Python2
    setup(name='pbdlib',
          version='0.1',
          description='Programming by Demonstration module for Python',
          url='',
          author='Emmanuel Pignat',
          author_email='emmanuel.pignat@idiap.ch',
          license='MIT',
          packages=find_packages(),
          install_requires=['numpy', 'scipy', 'matplotlib', 'sklearn', 'dtw', 'jupyter', 'enum', 'termcolor'],
          zip_safe=False)
else:
    # Setup for Python3
    setup(name='pbdlib',
          version='0.1',
          description='Programming by Demonstration module for Python',
          url='',
          author='Emmanuel Pignat',
          author_email='emmanuel.pignat@idiap.ch',
          license='MIT',
          packages=find_packages(),
          install_requires=['numpy', 'scipy', 'matplotlib', 'sklearn', 'dtw', 'jupyter', 'termcolor'],
          zip_safe=False)
