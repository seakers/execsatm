from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='ExecSATM',
    version='1.0.0',
    description='Executable Science and Applications Traceability Matrix for Earth-observation missions',
    author='SEAK Lab',
    author_email='aguilaraj15@tamu.edu',
    packages=['execsatm'],
    scripts=[],
    install_requires=['setuptools', 'numpy', 'pyparsing'] 
)
