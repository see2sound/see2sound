import os.path

from setuptools import find_packages, setup


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="see2sound",
    version=get_version("see2sound/version.py"),
    description="SEE-2-SOUND: Zero-Shot Spatial Environment-to-Spatial Sound",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/see2sound/see2sound",
    author="Rishit Dagli, Shivesh Prakash",
    author_email="rishit.dagli@gmail.com",
    install_requires=[],
    extras_require={
        "dev": ["check-manifest", "twine", "ruff"],
    },
)
