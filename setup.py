from setuptools import setup

with open('requirements.txt') as f:
  requirements = f.read().splitlines()

setup(
  name='elementarise_image',
  version='0.1.0',
  author='Martin Douša',
  author_email='martindousa186@gmail.com',
  packages=['elementarise_image'],
  url='http://pypi.python.org/pypi/elementarise_image/',
  license='LICENSE',
  description='Generate elementarised image',
  long_description=open('README.md').read(),
  install_requires=requirements,
)
