from pkg_resources import parse_requirements
from setuptools import setup

with open("VERSION") as f:
    version = f.read().strip()

with open("requirements.txt") as f:
    requirements = [str(req) for req in parse_requirements(f.read())]

setup(name="PARCSANN", python_requires=">=3.11", version=version, install_requires=requirements)
