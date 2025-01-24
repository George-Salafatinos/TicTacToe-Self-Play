from setuptools import setup, find_packages

setup(
    name="tictactoe-self-play",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "numpy",
        "torch",
        "matplotlib",
    ],
)