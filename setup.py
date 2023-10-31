import os
from glob import glob
from setuptools import setup

package_name = 'team5_final'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='michelangelo',
    maintainer_email='michelangelo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'get_object_range = team5_final.get_object_range:main',
            'navigate_maze = team5_final.navigate_maze:main',
            'classifier = team5_final.classifier:main'
        ],
    },
)
