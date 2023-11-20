from setuptools import setup


setup(
    name='catdog',
    description='a suite for training models that know cats from dogs',
    author='Emmanuel I. Obi',
    maintainer='Emmanuel I. Obi',
    maintainer_email='withtwoemms@gmail.com',
    url='https://github.com/withtwoemms/catdog',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    install_requires=[
        'kaggle==1.5.16',
        'numpy==1.26.2',
        'pillow==10.1.0',
        'scipy==1.11.4',
        'tensorflow==2.14.0',
    ],
    entry_points = {
        'console_scripts': [
            'train = catdog.train:execute_regimen',
            'predict = catdog.predict:make_prediction',
        ],
    }
)
