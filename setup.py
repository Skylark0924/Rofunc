from setuptools import setup, find_packages

setup(
    name="rofunc",
    version="0.0.0.9",
    description='Full-process robot learning from demonstration package',
    author="Junjia Liu",
    author_email="jjliu@mae.cuhk.edu.hk",
    url='https://github.com/Skylark0924/Rofunc',
    packages=find_packages(),
    package_data={"rofunc.data": ["*.txt", "*.npy"]},
    # include_package_data=True,
    install_requires=['matplotlib', 'pandas', 'tqdm', 'pillow', 'pytransform3d', 'tensorflow', 'numpy==1.21.6',
                      'nestle', 'omegaconf', 'hydra-core', 'opencv-python',],
                      # 'pbdlib @ https://github.com/Skylark0924/Rofunc/releases/download/v0.0.0.7.1/pbdlib-0.1-py3-none-any.whl',
                      # 'isaacgym @ https://github.com/Skylark0924/Rofunc/releases/download/v0.0.0.9/isaacgym-1.0rc4-py3-none-any.whl',
                      # 'pyzed @ https://github.com/Skylark0924/Rofunc/releases/download/v0.0.0.9/pyzed-3.7-cp37-cp37m-linux_x86_64.whl'],
                      # 'pbdlib @ file:///home/ubuntu/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/whl/pbdlib-0.1-py3-none-any.whl',
                      # 'isaacgym @ file:///home/ubuntu/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/whl/isaacgym-1.0rc4-py3-none-any.whl',
                      # 'pyzed @ file:///home/ubuntu/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/whl/pyzed-3.7-cp37-cp37m-linux_x86_64.whl'],
    python_requires=">=3.6,<3.11",
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
