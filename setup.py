import setuptools

setuptools.setup(
    name="RAscore",  # Replace with your own username
    version="2020.9",
    author="Reymond Group/Molecular AI AstraZeneca",
    author_email="amol.thakkar@dcb.unibe.ch",
    license="MIT",
    description="Computation of retrosynthetic accessibility from machine learening of CASP predictions",
    url="https://github.com/reymond-group/RAscore",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "rascore = RAScore.command_line_interface.py:main",
            "RAscore = RAScore.command_line_interface.py:main",
        ],
    },
)
