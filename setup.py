from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="rofunc",
    version="0.0.1.2",
    description='The Full Process Python Package for Robot Learning from Demonstration',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Junjia Liu",
    author_email="jjliu@mae.cuhk.edu.hk",
    url='https://github.com/Skylark0924/Rofunc',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['matplotlib', 'pandas', 'tqdm', 'pillow', 'pytransform3d', 'numpy', 'pynput',
                      'nestle', 'omegaconf', 'hydra-core', 'opencv-python', 'neurokit2', 'skrl', 'gdown', 'dm_tree',
                      'openpyxl', 'pytz', 'urdfpy', 'pin', 'shutup', 'elegantrl', 'ray[rllib]==2.2.0'],
    # 'pbdlib @ https://github.com/Skylark0924/Rofunc/releases/download/v0.0.0.7.1/pbdlib-0.1-py3-none-any.whl',
    # 'isaacgym @ https://github.com/Skylark0924/Rofunc/releases/download/v0.0.0.9/isaacgym-1.0rc4-py3-none-any.whl',
    # 'pyzed @ https://github.com/Skylark0924/Rofunc/releases/download/v0.0.0.9/pyzed-3.7-cp37-cp37m-linux_x86_64.whl'],
    python_requires=">3.6,<3.9",
    keywords=['robotics', 'learning from demonstration'],
    license='MIT',
    entry_points={
        'console_scripts': [
            'rf=rofunc._main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
