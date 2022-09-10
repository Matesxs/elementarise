from setuptools import setup, find_packages

def get_version() -> str:
  rel_path = "elementarise/__init__.py"
  with open(rel_path, "r") as fp:
    for line in fp.read().splitlines():
      if line.startswith("__version__"):
        delim = '"' if '"' in line else "'"
        return line.split(delim)[1]
  raise RuntimeError("Unable to find version string.")

with open('requirements.txt') as f:
  requirements = f.read().splitlines()

setup(
  name='elementarise',
  version=get_version(),
  author='Martin Douša',
  author_email='martindousa186@gmail.com',
  packages=find_packages(),
  url='https://github.com/Matesxs/elementarise',
  license='MIT',
  classifiers=[
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: End Users/Desktop",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.7",
    "Topic :: Multimedia",
    "Topic :: Utilities"
  ],
  description='Generate elementarised image',
  long_description_content_type='text/markdown',
  long_description=open('README.md').read(),
  install_requires=requirements,
)
