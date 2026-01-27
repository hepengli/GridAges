from pathlib import Path
from setuptools import setup, find_packages

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="gridages",  # PyPI + pip install gridages
    version="0.1.0",
    description="GridAges (AGES): agent-centric power grid simulator for RL and MARL.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/hepengli/GridAges",
    project_urls={
        "Source": "https://github.com/hepengli/GridAges",
        "Issues": "https://github.com/hepengli/GridAges/issues",
    },
    author="Hepeng Li",
    author_email="hepeng.li@maine.edu",
    license="MIT",
    license_files=("LICENSE",),
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "docs", "examples")),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "gymnasium>=0.29",
        "pandapower",
        "numpy",
        "pandas",
        "pettingzoo",  # now first-class
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff",
            "black",
            "mypy",
            "build",
            "twine",
        ],
    },
    keywords=[
        "power-grid",
        "reinforcement-learning",
        "multi-agent",
        "pettingzoo",
        "pandapower",
        "simulation",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
