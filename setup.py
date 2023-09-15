from setuptools import find_packages, setup

setup(
    name="vec2text",
    version="0.0.5",
    description="convert embedding vectors back to text",
    author="Jack Morris",
    author_email="jxm3@cornell.edu",
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines()
    # install_requires=[],
)
