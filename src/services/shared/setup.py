from setuptools import setup, find_packages

setup(
    name="specs-schemas",
    version="1.0.0",
    description="Shared message schemas for the spec-driven code generation system",
    author="Your Organization",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    # Include package data
    include_package_data=True,
    # Add entry points if needed
    entry_points={},
    # Add URLs
    project_urls={
        "Source": "https://github.com/yourusername/specs-schemas",
        "Documentation": "https://github.com/yourusername/specs-schemas/docs",
    },
)