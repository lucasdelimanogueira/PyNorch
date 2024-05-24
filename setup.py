import setuptools
from setuptools.command.install import install
import subprocess

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()



apt_dependencies = [
    'nvidia-cuda-toolkit'
]

apt_get_dependencies = [
    'mpi',
    'libopenmpi-dev'
]

apt_nccl = [
    'libnccl2=2.21.5-1+cuda12.2',
    'libnccl-dev=2.21.5-1+cuda12.2'
]

subprocess.check_call(['sudo', 'apt-get', 'update'])

for package in apt_dependencies:
    subprocess.check_call(['sudo', 'apt', 'install', package])

for package in apt_get_dependencies:
    subprocess.check_call(['sudo', 'apt-get', 'install', '-y', package])

subprocess.check_call(['wget', 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb'])
subprocess.check_call(['sudo', 'dpkg', '-i', 'cuda-keyring_1.0-1_all.deb'])

subprocess.check_call(['sudo', 'apt-mark', 'unhold', 'libnccl-dev', 'libnccl2'])

for package in apt_nccl:
    subprocess.check_call(['sudo', 'apt', 'install', package])

subprocess.check_call(['rm', 'cuda-keyring_1.0-1_all.deb'])

class CustomInstall(install):
    def run(self):
        subprocess.call(['make', '-C', 'build'])
        install.run(self)


setuptools.setup(
    name = "norch",
    version = "0.0.4",
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
    packages = setuptools.find_packages(),
    package_data={'norch': ['csrc/*', 'libtensor.so']},
    cmdclass={
        'install': CustomInstall,
    },
    python_requires = ">=3.6"
)
