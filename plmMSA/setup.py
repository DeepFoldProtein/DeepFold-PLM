from setuptools import find_packages, setup

setup(
    name="plmMSA",
    version="0.1.0",
    author="Hanjin Bae",
    author_email="iwdhanjin@gmail.com",
    description=".",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
