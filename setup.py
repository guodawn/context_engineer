"""Setup script for ContextEngineer package."""

from setuptools import setup, find_packages
import pathlib

# Get the long description from the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else "ContextEngineer: A system for optimizing context management in large language models."

setup(
    name="context-engineer",
    version="0.1.0",
    description="A system for optimizing context management in large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/context-engineer",
    author="ContextEngineer Team",
    author_email="team@contextengineer.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="llm, context, optimization, token-budgeting, ai",
    package_dir={"": "."},
    packages=find_packages(),
    python_requires=">=3.8, <4",
    install_requires=[
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "tiktoken": [
            "tiktoken>=0.4.0",
        ],
        "all": [
            "tiktoken>=0.4.0",
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "context-engineer=context_engineer.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/context-engineer/issues",
        "Source": "https://github.com/your-username/context-engineer",
        "Documentation": "https://context-engineer.readthedocs.io/",
    },
)