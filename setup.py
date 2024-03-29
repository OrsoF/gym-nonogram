import sys
import os
from setuptools import setup, find_packages

CURRENT_PYTHON = sys.version_info[:2]
MIN_PYTHON = (3, 6)

if CURRENT_PYTHON < MIN_PYTHON:
    sys.stderr.write("""
        ============================
        Unsupported Python Version
        ============================
        Python {}.{} is unsupported. Please use a version newer than Python {}.{}.
    """.format(*CURRENT_PYTHON, *MIN_PYTHON))
    sys.exit(1)

with open('requirements.txt', 'r') as f:
    install_requires = f.readlines()

with open('README.md') as f:
    README = f.read()

setup(name='gym-nonogram',
      description='OpenAI Gym Environment for Nonogram',
      long_description=README,
      long_description_content_type='text/x-rst',
      version=0.1,
      url='https://github.com/OrsoF/gym-nonogram',
      author='Orso Forghieri',
#      license='',
      packages=find_packages(),
      install_requires=install_requires,
      extras_require={},
)