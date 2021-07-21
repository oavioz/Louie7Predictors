from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="vector_creator",
    version="1.0.0",
    description="Project template",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url="https://",
    author="Louie7.ai",
    author_email="shmuel@louie7.ai",
    keywords="core package",
    license="MIT",
    packages=[
        'vector_creator',
        'vector_creator.preprocess',
        'vector_creator.raw_to_df',
        'vector_creator.score_vectors',
        'vector_creator.stats_models'
    ],
    install_requires=['pandas', 'numpy', 'statsmodels', 'sklearn', 'scipy', 'geopy'],
    include_package_data=True,
    zip_safe=False
)