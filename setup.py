from setuptools import setup, find_packages

setup(
    name="rofunc",
    version="0.0.0.7",
    description='Useful functions for robot experiments',
    author="skylark",
    author_email="jjliu@mae.cuhk.edu.hk",
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'pandas', 'tqdm', 'pillow'],
    python_requires='>=3.6',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
