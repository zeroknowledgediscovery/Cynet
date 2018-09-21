from setuptools import setup
from codecs import open
from os import path
# import subprocess as sp
import warnings

version = {}
with open("version.py") as fp:
    exec(fp.read(), version)

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# sp.call("wget https://repo.continuum.io/miniconda/Miniconda2-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh",\
#         shell=True)
# sp.call("bash ~/miniconda.sh -b -p $HOME/miniconda", shell=True)
# sp.call('export PATH="$HOME/miniconda/bin:$PATH"', shell=True)
# sp.call('conda install -c -y conda-forge cartopy', shell=True)



warnings.warn("WARNING: Please ensure cartopy is installed - the viscynet library depends on cartopy.\
 Due to failing builds of cartopy, this package cannot autoinstall the cartopy dependency.\
 See https://scitools.org.uk/cartopy/docs/latest/installing.html#id2 for instructions.")



setup(
    name='cynet',
    author='zed.uchicago.edu',
    author_email='ishanu@uchicago.edu',
    version = str(version['__version__']),
    packages=['cynet','viscynet','bokeh_pipe'],
    scripts=['bin/print_help.py'],
    url='https://github.com/zeroknowledgediscovery/',
    license='LICENSE.txt',
    description='Learning Point Processes Using Deep Granger Nets',
    keywords=['spatial', 'temporal', 'inference', 'statistical', 'causality'],
    download_url='https://github.com/zeroknowledgediscovery/Cynet/archive/'+str(version['__version__'])+'.tar.gz',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    install_requires=["numpy >= 1.6","pandas == 0.22.0","matplotlib >= 2.0.2","scipy >= 0.18.1", \
    "tqdm >= 4.11.2","seaborn >= 0.8.0","sodapy >= 1.4.6","bokeh >= 0.12.14","pyproj >= 1.9.5.1",\
    "pyshp >= 1.2.12","haversine >= 0.4.5","joblib >= 0.12.2"],
    python_requires='==2.7.*',
    classifiers=[\
    'Development Status :: 4 - Beta',
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Software Development :: Libraries",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 2.7"],
    data_files=[('random/',['bin/print_help.py'])],
    include_package_data=True
)
