from setuptools import setup
import os
from glob import glob

package_name = 'start_reinforcement_learning'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name + '.mappo_algorithm', package_name + '.env_logic', package_name + '.maddpg_algorithm'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='unruly',
    maintainer_email='unruly@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_maddpg = start_reinforcement_learning.maddpg_main:main',
            'run_mappo = start_reinforcement_learning.mappo_main:main',
            'run_mappo_evaluate = start_reinforcement_learning.mappo_evaluate:main',
            'run_maddpg_evaluate = start_reinforcement_learning.maddpg_evaluate:main',
            'run_algorithm_comparison = start_reinforcement_learning.run_algorithm_comparison:main',
        ],
    },
)
