from setuptools import setup, find_packages

setup(
    name="medclean",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
    ],
    author="Pusula Test Case",
    description="Reusable medical data preprocessing library",
    python_requires=">=3.8",
)
