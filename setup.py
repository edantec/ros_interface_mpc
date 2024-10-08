from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ros_interface_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='edantec',
    maintainer_email='ewen.dantec@inria.fr',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker_python = python.publisher:main',
            'listener_python = python.subscriber:main',
            'state_publisher = python.state_publisher:main'
        ],
    },
)
