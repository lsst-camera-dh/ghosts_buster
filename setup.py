from setuptools import setup

setup(
    name="ghost_buster",
    author="Dimitri Buffat",
    author_email="dimitri.buffat@etu.univ-lyon1.fr",
    url = "https://github.com/dbuffat/lsst_ghost_buster",
    packages=["ghost_buster"],
    description="Analysis of Rubin telescope ghost images",
    setup_requires=['setuptools_scm'],
    long_description=open("README.md").read(),
    package_data={"": ["README.md"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Beta",
        "License :: OSI Approved :: GPL License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["batoid>=0.8",
                      "matplotlib",
                      "numpy",
                      "pandas",
                      "pyarrow",
                      "setuptools_scm"]
)
