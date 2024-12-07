from setuptools import setup, find_packages

setup(
    name='emotion-predictor',
    version='1.0.0',
    author='Your Name',
    author_email='your_email@example.com',
    description='Emotion prediction package using pre-trained models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your_github/emotion-predictor',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'scikit-learn>=0.24.0',
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
