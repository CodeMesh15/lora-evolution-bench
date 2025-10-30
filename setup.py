"""
Efficient Fine-Tuning Arena
A comprehensive benchmarking suite for parameter-efficient fine-tuning methods
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Core dependencies
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "transformers>=4.38.0",
    "peft>=0.8.0",
    "bitsandbytes>=0.41.0",
    "accelerate>=0.25.0",
    "datasets>=2.16.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
    "tqdm>=4.65.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
    "safetensors>=0.4.0",
    "einops>=0.7.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    # ReFT-specific dependencies
    "reft": [
        "pyreft>=0.0.8",
        "pyvene>=0.1.0",
        "matplotlib>=3.7.0",
        "plotnine>=0.12.0",
    ],
    
    # Development dependencies
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "pre-commit>=3.3.0",
        "ipython>=8.12.0",
        "ipykernel>=6.25.0",
    ],
    
    # Benchmarking dependencies
    "benchmark": [
        "evaluate>=0.4.0",
        "scikit-learn>=1.3.0",
        "seaborn>=0.12.0",
        "matplotlib>=3.7.0",
        "psutil>=5.9.0",
        "gputil>=1.4.0",
        "py3nvml>=0.2.7",
    ],
    
    # Experiment tracking
    "tracking": [
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        "mlflow>=2.8.0",
    ],
    
    # Distributed training
    "distributed": [
        "deepspeed>=0.12.0",
        "mpi4py>=3.1.4",
    ],
    
    # Documentation
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "sphinx-autodoc-typehints>=1.24.0",
        "myst-parser>=2.0.0",
    ],
    
    # Jupyter notebook support
    "notebooks": [
        "jupyter>=1.0.0",
        "notebook>=7.0.0",
        "jupyterlab>=4.0.0",
        "ipywidgets>=8.1.0",
    ],
}

# Add 'all' option to install everything
EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

# Add 'full' option (all except distributed which can be tricky)
EXTRAS_REQUIRE["full"] = list(set(
    EXTRAS_REQUIRE["reft"] + 
    EXTRAS_REQUIRE["benchmark"] + 
    EXTRAS_REQUIRE["tracking"] + 
    EXTRAS_REQUIRE["notebooks"]
))

setup(
    name="efficient-finetuning-arena",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive benchmarking suite for parameter-efficient fine-tuning methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/efficient-finetuning-arena",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/efficient-finetuning-arena/issues",
        "Documentation": "https://github.com/yourusername/efficient-finetuning-arena/docs",
        "Source Code": "https://github.com/yourusername/efficient-finetuning-arena",
    },
    packages=find_packages(
        include=["eft_arena", "eft_arena.*"],
        exclude=["tests", "tests.*", "docs", "docs.*", "notebooks", "notebooks.*"]
    ),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "eft-train=eft_arena.cli.train:main",
            "eft-benchmark=eft_arena.cli.benchmark:main",
            "eft-evaluate=eft_arena.cli.evaluate:main",
            "eft-convert=eft_arena.cli.convert:main",
        ],
    },
    include_package_data=True,
    package_data={
        "eft_arena": [
            "configs/*.yaml",
            "configs/*.json",
            "data/*.json",
        ],
    },
    keywords=[
        "deep-learning",
        "machine-learning",
        "nlp",
        "llm",
        "fine-tuning",
        "lora",
        "qlora",
        "dora",
        "reft",
        "peft",
        "parameter-efficient",
        "transformers",
        "pytorch",
    ],
    zip_safe=False,
)
