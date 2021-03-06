from setuptools import setup, find_packages
import os


# extract version
with open(os.path.join(os.path.dirname(__file__),
          "pypesto", "version.py")) as f:
    version = f.read().split('\n')[0].split('=')[-1].strip(' ').strip('"')


# read a file
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# project metadata
setup(name='pypesto',
      version=version,
      description="python Parameter EStimation TOolbox",
      long_description=read('README.md'),
      author="The pyPESTO developers",
      author_email="yannik.schaelte@gmail.com",
      url="https://github.com/icb-dcm/pypesto",
      packages=find_packages(exclude=["doc*", "test*"]),
      install_requires=['numpy>=1.15.1',
                        'scipy>=1.1.0',
                        'pandas>=0.23.4',
                        'matplotlib>=2.2.3'],
      extras_require={'amici': ['amici>=0.7.10']}
      )
