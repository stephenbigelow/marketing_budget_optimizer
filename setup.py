from setuptools import setup, find_packages

setup(
    name="marketing_budget_optimizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
    ],
    author="Stephen Bigelow",
    author_email="your.email@example.com",
    description="A Python package for optimizing marketing budgets using response curves",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/marketing_budget_optimizer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 