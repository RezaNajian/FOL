from setuptools import setup, find_packages

setup(
    name='fol',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'meshio',
        'jax',
        'jaxopt',
        'gmsh'
    ],
    author='FOL team',
    author_email='r.najian@hotmail.com',
    description='Bridging Neural Operators and Numerical Methods',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RezaNajian/FOL',  
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires='>=3.10',
)
