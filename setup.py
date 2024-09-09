from setuptools import setup

setup(
    name="h3h",
    version="0.1.0",
    packages=["h3h"],
    include_package_data=False,
    python_requires=">=3.11",
    install_requires=[
        "geopandas",
        "h3",
        "h3pandas",
        "rioxarray",
        "statsmodels",
        "scikit-learn",
        "toml",
        "tqdm",
        "contextly",
        "seaborn",
    ],
)
