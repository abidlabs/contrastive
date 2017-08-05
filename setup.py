try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name = 'contrastive',
    version = '1.0.0',
    description = 'Python library for performing unsupervised learning (e.g. PCA) in contrastive settings, where one is interested in patterns that exist one dataset, but not the other',
    author = 'Abubakar Abid',
    author_email = 'a12d@stanford.edu',
    url = 'https://github.com/abidlabs/contrastive', 
    download_url = '',
    packages=['contrastive'],
    keywords = ['unsupervised', 'contrastive', 'learning','PCA'],
    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'matplotlib',
    ],
)