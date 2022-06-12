from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="alphascope",
    version="0.0.2",
    description="gen purpose quant fin library",
    license="MIT",
    url="https://github.com/atkrishnan/alphascope",
    author="Atheesh Krishnan",
    author_email="atheesh.krishnan@outlook.com",
    # py_modules=["helloworld"],
    # package_dir={'': 'alphascope'},
    packages=["alphascope"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Natural Language :: English",
    ],
    keywords="computational finance optimization quant trading investing portfolio arbitrage",
    install_requires=["numpy", "pandas", "scipy", "cvxpy", "statsmodels"],
    extras_requires={
        "dev": [
            "pytest>=3.7"
        ]
    },
    python_requires=">=3.5",
    long_description=long_description,
    long_description_content_type="text/markdown"
)
