from setuptools import setup
from codecs import open
from os import path

version = {}
with open("version.py") as fp:
    exec(fp.read(), version)

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cynet',
    author='zed.uchicago.edu',
    author_email='ishanu@uchicago.edu',
    version = str(version['__version__']),
    packages=['cynet','viscynet','bokeh_pipe'],
    scripts=['bin/print_help.py','bin/read_raw_log.py'],
    url='https://github.com/zeroknowledgediscovery/',
    license='LICENSE.txt',
    description='Spatio temporal analysis for inferrence of statistical causality using XGenESeSS',
    keywords=['spatial', 'temporal', 'inference', 'statistical', 'causality'],
    download_url='https://github.com/zeroknowledgediscovery/Cynet/archive/'+str(version['__version__'])+'.tar.gz',
    long_description=open('README.rst').read(),
    install_requires=["numpy >= 1.13.1","pandas >= 0.19.0","matplotlib >= 2.0.2","scipy >= 0.18.1", \
    "tqdm >= 4.11.2","seaborn >= 0.8.0", "sodapy >= 1.4.6", "bokeh >= 0.12.14", "pyproj >= 1.9.5.1",
    "pyshp >= 1.2.12"],
    python_requires='==2.7.*',
    classifiers=[
    'Development Status :: 4 - Beta',
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Software Development :: Libraries",
    "License :: OSI Approved :: MIT License",
    'Programming Language :: Python :: 2.7'],
)
