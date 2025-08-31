# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ormax",
    version="1.2.3",
    author="Shayan Heidari",
    author_email="shayanheidari01@gmail.com",
    description="High-performance async ORM for all major databases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shayanheidari01/ormax",
    project_urls={
        "Bug Tracker": "https://github.com/shayanheidari01/ormax/issues",
        "Documentation": "https://github.com/shayanheidari01/ormax#readme",
        "Source Code": "https://github.com/shayanheidari01/ormax",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: AsyncIO",
    ],
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.7",
    install_requires=[],
    extras_require={
        "mysql": ["aiomysql>=0.1.14"],
        "postgresql": ["asyncpg>=0.27.0"],
        "sqlite": ["aiosqlite>=0.19.0"],
        "mssql": ["aioodbc>=0.17.0"],
        "oracle": ["async-oracledb>=1.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.950",
        ],
        "all": [
            "aiomysql>=0.1.14",
            "asyncpg>=0.27.0", 
            "aiosqlite>=0.19.0",
            "aioodbc>=0.17.0",
            "async-oracledb>=1.0.0",
        ]
    },
    keywords=[
        "orm", "async", "asyncio", "database", "sql", "mysql", 
        "postgresql", "sqlite", "mariadb", "database-orm",
        "mssql", "sqlserver", "oracle", "aurora"
    ],
    license="MIT",
    zip_safe=False,
    include_package_data=True,
)