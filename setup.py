try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name = 'contrastive',
    version = '1.1.0',
    description = 'Python library for performing unsupervised learning (e.g. PCA) in contrastive settings, where one is interested in finding directions and patterns that exist one dataset, but not the other',
    author = 'Abubakar Abid',
    author_email = 'a12d@stanford.edu',
    url = 'https://github.com/abidlabs/contrastive',
    # download_url = 'https://github.com/abidlabs/contrastive/archive/0.1.tar.gz',
    packages=['contrastive'],
    keywords = ['unsupervised', 'contrastive', 'learning','PCA'],
    install_requires=[
        'numpy',
        'sklearn',
        'matplotlib',
    ],
)
