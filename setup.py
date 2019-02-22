from setuptools import setup, find_packages

setup(name='cytofpy',
      version='0.2',
      description='Cytof variational inference model implemented using PyTorch',
      url='http://github.com/cytofpy',
      author='Arthur Lui',
      author_email='luiarthur@gmail.com',
      license='MIT',
      packages=find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=['torch', 'np', 'matplotlib'],
      zip_safe=False)
