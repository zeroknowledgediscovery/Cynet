from setuptools import setup

setup(
    name='cynet',
    version='1.0.1',
    author='zed.uchicago.edu',
    author_email='ishanu@uchicago.edu',
    packages=['cynet'],
    scripts=['bin/print_help.py','bin/process_log1.py','bin/process_models.py','bin/read_raw_log1.py',\
    'bin/script_test.py','bin/script11.py'],
    url='https://github.com/zeroknowledgediscovery/spin_',
    license='LICENSE.txt',
    description='Spatio temporal analysis for inferrence of statistical causality using XGenESeSS',
    keywords=['spatial','temporal','inference','statistical','causality'],
    download_url='https://github.com/zeroknowledgediscovery/Cynet/archive/1.0.1.tar.gz',
    long_description=open('README.rst').read(),
    install_requires=["numpy >= 1.13.1","pandas >= 0.19.0","matplotlib >= 2.0.2","scipy >= 0.18.1", \
    "tqdm >= 4.11.2","seaborn >= 0.8.0","pickle >= 1.71", "sodapy >= 1.4.6"],
    classifiers=[],
)
