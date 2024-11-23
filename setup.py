from setuptools import setup, find_packages

setup(
        name='URF',
        version='2024.0',
        author='Mohammad Sahil',
        author_email='msahil@tifrh.res.in',
        description=open('README.md','r').read().strip().split('\n')[0][3:],
        long_description=open('README.md','r').read().strip().split('\n')[4],
        long_description_content_type="text/markdown",
        packages=find_packages(),
        install_requires=[
            'numpy>=1.25,<2.0',
            'scikit-learn>=1.3',
            'scipy>=1.11',
            'numba>=0.59',
            'fastcluster>=1.2',
            'matplotlib>=3.7',
            'tqdm>=4.63',
            'joblib>=1.3'
            ],
        include_package_data=False,
        url={
            'code':'',
            'publication':'',
            'data':''
        },
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            ],
        python_requires='>=3.9',
)


