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
    version="0.0.2.6",
    description='Rofunc: The Full Process Python Package for Robot Learning from Demonstration and Robot Manipulation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Junjia Liu",
    author_email="jjliu@mae.cuhk.edu.hk",
    url='https://github.com/Skylark0924/Rofunc',
    packages=find_packages(),
    include_package_data=True,
    extras_require=extras,
    install_requires=["shutup",
                      "setuptools==57.4.0",
                      "omegaconf==2.3.0",
                      'opencv-python==4.7.0.72',
                      'tqdm==4.67.1',
                      'pandas==2.0.3',
                      'hydra-core==1.3.2',
                      'trimesh==4.5.3',
                      'lxml==5.3.0',
                      "pytorch_kinematics==0.7.2",
                      'nestle==0.2.0',
                      'gdown==5.2.0',
                      "transformations==2022.9.26",
                      'gym>=0.26.2',
                      "numpy<=1.23.0",
                      'wandb==0.18.7',
                      "gymnasium==1.0.0",
                      "tensorboard==2.14.0",
                      "transformers==4.46.3"
                      ],
    python_requires=">=3.7,<3.11",
    keywords=['robotics', 'robot learning', 'learning from demonstration', 'reinforcement learning',
              'robot manipulation'],
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
