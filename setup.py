from setuptools import setup, find_packages


setup(
    name='virtual_nematode',
    version='0.0.1',
    packages=[package for package in find_packages() if package.startswith('virtual_nematode')],
    install_requires=[],
    python_requires=">=3.8",
)
