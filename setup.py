#!/usr/bin/env python3

from setuptools import setup
from setuptools import find_packages

setup(
    name="essaysense",
    version="0.0.4",
    author="Zilong Liang, Jiancong Gao",
    author_email="15300180026@fudan.edu.cn",
    url="https://github.com/deltaquincy/essaysense",
    description="EssaySense is an NLP project on Automated Essay Scoring, based on neural network technologies.",
    python_requires=">=3",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorflow",
        "nltk",
        "click"
    ]
)
