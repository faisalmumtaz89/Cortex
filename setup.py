"""Setup script for Cortex."""

import sys
from pathlib import Path
from setuptools import setup, find_packages

# Check Python version
if sys.version_info < (3, 11):
    print("Error: Cortex requires Python 3.11 or later")
    sys.exit(1)

# Read README
README = Path("README.md").read_text() if Path("README.md").exists() else ""

def read_requirements():
    """Read requirements from requirements.txt."""
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        with open(requirements_file) as f:
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith("#")
            ]
    return []

setup(
    name="cortex-llm",
    version="1.0.10",
    author="Cortex Development Team",
    description="GPU-Accelerated LLM Terminal for Apple Silicon",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/faisalmumtaz/Cortex",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS",
        "Environment :: Console",
        "Environment :: GPU",
    ],
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.8.0",
        ],
        "optional": [
            "sentencepiece>=0.1.99",
            "auto-gptq>=0.7.0",
            "autoawq>=0.2.0",
            "bitsandbytes>=0.41.0",
            "optimum>=1.16.0",
            "torchvision>=0.16.0",
            "torchaudio>=2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cortex=cortex.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cortex": ["config.yaml"],
    },
    zip_safe=False,
    platforms=["darwin"],
    keywords=[
        "llm", "gpu", "metal", "mps", "apple-silicon",
        "ai", "machine-learning", "terminal", "mlx", "pytorch",
    ],
)
