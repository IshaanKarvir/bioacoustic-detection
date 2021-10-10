import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bioacoustic-detection",
    version="0.0.1",
    author="Jackson Waschura",
    author_email="jackson.waschura@gmail.com",
    description="A pre-trained model and API for detecting humpback whale " \
        "vocalizations in hydrophone recordings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jackson-waschura/bioacoustic-detection",
    project_urls={
        "Bug Tracker":
            "https://github.com/jackson-waschura/bioacoustic-detection/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)