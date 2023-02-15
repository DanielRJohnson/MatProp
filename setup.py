import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="matprop",
    version="0.0.1",
    license='MIT',
    author="Daniel Johnson",
    author_email="danielrjohnsonprofessional@gmail.com",
    description="A small PyTorch-like backpropagation engine and neural network framework defined with "
                "autograd-supported matrix operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danielrjohnson/matprop",
    packages=setuptools.find_packages(),
    keywords="backpropagation neural network autograd",
    install_requires=[
          'numpy',
      ],
)
