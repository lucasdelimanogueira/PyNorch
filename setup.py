import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "norch",
    version = "0.0.1",
    scripts=['install.sh'],
    author = "Lucas de Lima",
    author_email = "nogueiralucasdelima@gmail.com",
    description = "A deep learning framework",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/lucasdelimanogueira/PyNorch",
    project_urls = {
        "Bug Tracker": "https://github.com/lucasdelimanogueira/PyNorch/issues",
        "Repository": "https://github.com/lucasdelimanogueira/PyNorch"
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "norch"},
    packages = setuptools.find_packages(where="norch"),
    python_requires = ">=3.6"
)
