import codecs
import sys
import os
from setuptools import setup, find_packages
from glob import glob

def gen_data_files(dirs):
    results = []
    for datadir in dirs:
        for p, _, files in os.walk(datadir):
            results.extend((p, [os.path.join(p, f)]) for f in files)
    print(results)
    return results

setup(
    name='bundle',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/mfojtak/bundle',
    license='MIT',
    author='mfojtak',
    author_email='mfojtak@seznam.cz',
    description='bundle',
    install_requires=["Jinja2", "pipreqs"],
    scripts=['bundlectl'],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    data_files=gen_data_files(["templates"]),
)