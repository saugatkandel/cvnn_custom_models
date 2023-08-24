from setuptools import setup

setup(
    name='cvnn_custom',
    version='0.0.1',
    author='Saugat Kandel',
    author_email='saugat.kandel@u.northwestern.edu',
    packages=['cvnn_custom'],
    #packages=['optimizers', 'tests', 'benchmarks', 'examples'],
    #package_data={'data': ['*.png']},
    scripts=[],
    description='Custom models for complex valued neural nets',
    install_requires=[
        "tensorflow",
        "cvnn"
    ],
)
