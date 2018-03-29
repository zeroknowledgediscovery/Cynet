from setuptools import setup

setup(
    name='spin',
    version='1.0.0',
    author='zed.uchicago.edu',
    author_email='ishanu@uchicago.edu',
    packages=['spin', 'spin.test'],
    scripts=['bin/print_help.py','bin/process_log1.py','bin/process_models.py','bin/read_raw_log1.py','bin/script_test.py'\
            'bin/script11.py','bin/script2.py'],
    url='https://github.com/zeroknowledgediscovery/spin_',
    license='LICENSE.txt',
    description='Spatio temporal analysis for inferrence of statistical causality using XGenESeSS',
    keywords=['spatial','temporal','inference','statistical','causality'],
    download_url='https://github.com/zeroknowledgediscovery/spin_/archive/1.0.0.tar.gz',
    long_description=open('README.md').read(),
    install_requires=["numpy == 1.13.1","pandas == 0.19.0","matplotlib >= 2.0.2","scipy >= 0.18.1"],
    classifiers=[],
)
