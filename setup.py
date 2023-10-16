from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Environment-specific dependencies.
extras = {
    "baselines": ["skrl==0.10.2", "ray[rllib]==2.2.0", "stable-baselines3==1.8.0", "rl-games==1.6.0",
                  "mujoco_py==2.1.2.14", "gym[all]==0.26.2", "gymnasium[all]==0.28.1", "mujoco-py<2.2,>=2.1"],
}

setup(
    name="rofunc",
    version="0.0.2.5",
    description='Rofunc: The Full Process Python Package for Robot Learning from Demonstration and Robot Manipulation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Junjia Liu",
    author_email="jjliu@mae.cuhk.edu.hk",
    url='https://github.com/Skylark0924/Rofunc',
    packages=find_packages(exclude=["others"]),
    include_package_data=True,
    extras_require=extras,
    install_requires=['cython==3.0.0a10',  # for mujoco_py
                      'setuptools==63.2.0',
                      'pandas',
                      'tqdm==4.65.0',
                      'pillow==9.5.0',
                      'nestle==0.2.0',
                      'omegaconf==2.3.0',
                      'hydra-core==1.3.2',
                      'opencv-python==4.7.0.72',
                      'neurokit2==0.2.4',
                      'gdown==4.7.1',
                      'pytz==2023.3',
                      'shutup==0.2.0',
                      'numpy<=1.23.0',
                      'matplotlib',
                      'open3d==0.17.0',
                      'transformers==4.30.1',
                      'gym>=0.26.2',
                      'gymnasium>=0.28.1',
                      'tensorboard',
                      'networkx',
                      'dgl'],
    python_requires=">=3.7,<3.9",
    keywords=['robotics', 'learning from demonstration', 'reinforcement learning', 'robot manipulation'],
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
