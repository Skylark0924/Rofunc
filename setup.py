from setuptools import setup, find_packages

setup(
    name="rofunc",
    version="0.0.0.4",
    author="skylark",
    author_email="jjliu@mae.cuhk.edu.hk",
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'pandas', 'tqdm', 'pillow'],
    python_requires='>=3.6',
)
