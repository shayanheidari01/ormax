from pathlib import Path
import re

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
INIT = (ROOT / "ormax" / "__init__.py").read_text(encoding="utf-8")
VERSION = re.search(r'^__version__ = "([^"]+)"', INIT, re.MULTILINE).group(1)

setup(
    name="ormax",
    version=VERSION,
    author="Shayan Heidari",
    author_email="shayanheidari01@gmail.com",
    description="Lightweight asynchronous ORM with multi-database support",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/shayanheidari01/ormax",
    project_urls={
        "Bug Tracker": "https://github.com/shayanheidari01/ormax/issues",
        "Documentation": "https://shayanheidari01.github.io/ormax/",
        "Source": "https://github.com/shayanheidari01/ormax",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Framework :: AsyncIO",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    packages=find_packages(exclude=("tests*", "examples*")),
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "sqlite": ["aiosqlite>=0.19"],
        "mysql": ["aiomysql>=0.1.14"],
        "postgresql": ["asyncpg>=0.27"],
        "mssql": ["aioodbc>=0.17"],
        "oracle": ["oracledb>=2.0"],
        "all": [
            "aiosqlite>=0.19",
            "aiomysql>=0.1.14",
            "asyncpg>=0.27",
            "aioodbc>=0.17",
            "oracledb>=2.0",
        ],
        "dev": [
            "aiosqlite>=0.19",
            "pytest>=7",
            "pytest-asyncio>=0.20",
            "black>=24",
            "flake8>=6",
            "mypy>=1.8",
            "build>=1",
            "twine>=5",
        ],
    },
    keywords=[
        "orm",
        "async",
        "asyncio",
        "database",
        "sql",
        "mysql",
        "postgresql",
        "sqlite",
        "mariadb",
        "mssql",
        "oracle",
        "aurora",
    ],
    license="MIT",
    include_package_data=True,
    package_data={"ormax": ["py.typed"]},
    zip_safe=False,
)
