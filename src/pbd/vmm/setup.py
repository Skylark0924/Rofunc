from setuptools import setup, find_packages

# Setup:
setup(name='vmm',
      version='0.1',
      description='Variational mixture model library',
      url='',
      author='Anonymous',
      author_email='anonymous@somewhere.someletters',
      license='MIT',
      packages=find_packages(),
      install_requires = [
          'numpy','scipy','matplotlib', 'tensorflow_probability', 'ipykernel', 'pyyaml', 'jupyter', 'lxml'],
      zip_safe=False)
