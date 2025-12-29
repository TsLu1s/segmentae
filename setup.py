import setuptools
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="segmentae",
    version="1.5.10", 
    description="SegmentAE: A Python Library for Anomaly Detection Optimization",
    long_description=long_description,      
    long_description_content_type="text/markdown",
    url="https://github.com/TsLu1s/SegmentAE",
    author="Luís Fernando da Silva Santos",
    author_email="luisf_ssantos@hotmail.com",
    license="MIT",
    classifiers=[
        # Indicate who your project is intended for
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Customer Service",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Telecommunications Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",

        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",

        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    py_modules=["segmentae"],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},  
    keywords=[
        "python"
        "data science",
        "machine learning",
        "deep learning",
        "neural networks",
        "autoencoder",
        "clustering",
        "anomaly detection",
        "novelty detection"
        "fraud detection",
        "data preprocessing",
    ],           
    install_requires=open("requirements.txt").readlines(),
)
