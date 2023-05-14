from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='portfolioperformance',
    version='1.0.0',
    author='Siddharth Nand',
    author_email='snand233@gmail.com',
    
    url="https://github.com/sidnand/portfolioperformance",

    description="Tool to test the out-of-sample performance of portfolio optimization models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    packages=find_packages(),
    
    license="GNU",
    license_files=["LICENSE.txt"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
