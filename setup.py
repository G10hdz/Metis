from setuptools import setup

setup(
    name='metis',
    version='2.0.0',
    package_dir={'': 'src'},
    packages=[
        'src',
        'src.graph',
        'src.agents',
        'src.memory',
        'src.config',
        'src.utils',
        'src.telegram',
    ],
    python_requires='>=3.10',
)
