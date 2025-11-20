from setuptools import setup, find_packages

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
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=requirements,
    entry_points={
            'console_scripts': [
                'frmis-stitch=frmis_stitching.main:main', # <-- MUST CHANGE
            ],
        },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose an appropriate license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
