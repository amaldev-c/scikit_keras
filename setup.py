"""Package distribution configuration"""
from setuptools import setup,find_packages

setup(
    name='fire_ext',
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    package_data={'fire_ext_model': ['*.yml','dataset/*.arff']},
    install_requires=['scikit-learn','pandas','scipy','pyyaml','tensorflow']
)
