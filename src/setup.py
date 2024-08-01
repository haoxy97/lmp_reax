from setuptools import setup, find_packages

setup(
    name='lmp_reax',
    version='1.0',
    author='Hao Zhao',
    author_email='zhaohaoyx@stu.xjtu.edu.cn',
    description='Package for "Assessment on the Rings Cleavage Mechanism of Polycyclic Aromatic Hydrocarbons in Supercritical Water: A ReaxFF Molecular Dynamics Study"',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'openbabel'
    ],

    # Metadata
    long_description='This package contains scripts and utilities related to the study of Polycyclic Aromatic Hydrocarbons (PAHs) in supercritical water using ReaxFF molecular dynamics.',
    long_description_content_type='text/plain',
    url='https://github.com/haoxy97/lmp_reax',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
)
