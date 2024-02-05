from setuptools import setup
from pkg_resources import parse_requirements

with open('VERSION') as f:
    version = f.read().strip()

with open('requirements.txt') as f:
    requirements = [str(req) for req in parse_requirements(f.read())]

setup(name='PARCSANN', version=version, install_requires=requirements)