from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='frmis-stitching',
    version='1.0',
    author='Mohammed Sinad',
    author_email='sinadsiraj@gmail.com',
    description='A Python implementation of the FRMIS stitching algorithm.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mohdsinad/FRMIS-Python',
    py_modules=[
        "main",
        "stitch",
        "pairwise_alignment",
        "global_alignment",
    ],
    package_dir={"": "src"},
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'frmis-stitch=main:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose an appropriate license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
