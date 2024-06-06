import setuptools
from setuptools.command.install import install
import subprocess
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

apt_dependencies = [
    'nvidia-cuda-toolkit'
]

apt_get_dependencies = [
    'mpi',
    'libopenmpi-dev',
    'openmpi-common',
    'openmpi-bin'
]

apt_nccl = [
    'libnccl2=2.21.5-1+cuda12.2',
    'libnccl-dev=2.21.5-1+cuda12.2'
]

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            sys.stdout.flush()
    rc = process.poll()
    return rc

print("Updating package lists...")
sys.stdout.flush()
run_command(['sudo', 'apt-get', 'update'])

print("Installing apt dependencies...")
sys.stdout.flush()
for package in apt_dependencies:
    run_command(['sudo', 'apt', 'install', '-y', package])

print("Installing apt-get dependencies...")
sys.stdout.flush()
for package in apt_get_dependencies:
    run_command(['sudo', 'apt-get', 'install', '-y', package])

print("Downloading and installing CUDA keyring...")
sys.stdout.flush()
run_command(['wget', 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb'])
run_command(['sudo', 'dpkg', '-i', 'cuda-keyring_1.0-1_all.deb'])

print("Upgrading packages...")
sys.stdout.flush()
run_command(['sudo', 'apt-get', 'upgrade', '-y', '--allow-change-held-packages'])
run_command(['sudo', 'apt-mark', 'unhold', 'libnccl-dev', 'libnccl2'])

print("Installing NCCL packages...")
sys.stdout.flush()
for package in apt_nccl:
    run_command(['sudo', 'apt', 'install', '-y', package])

print("Cleaning up...")
sys.stdout.flush()
run_command(['rm', 'cuda-keyring_1.0-1_all.deb'])

class CustomInstall(install):
    def run(self):
        print("Running custom install steps...")
        sys.stdout.flush()
        run_command(['make', '-C', 'build'])
        install.run(self)

setuptools.setup(
    name="norch",
    version="0.0.7",
    author="Lucas de Lima",
    author_email="nogueiralucasdelima@gmail.com",
    description="A deep learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lucasdelimanogueira/PyNorch",
    project_urls={
        "Bug Tracker": "https://github.com/lucasdelimanogueira/PyNorch/issues",
        "Repository": "https://github.com/lucasdelimanogueira/PyNorch"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    package_data={'norch': ['csrc/*', 'libtensor.so']},
    cmdclass={
        'install': CustomInstall,
    },
    python_requires=">=3.6"
)
