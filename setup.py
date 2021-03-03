import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="RAscore",
        version="2021.03",
        description="Retrosynthetic Accessibility (RA) score learned from computer aided synthesis planning",
        long_description="file: README.md",
        url="https://github.com/reymond-group/RAscore",
        download_url="https://github.com/reymond-group/RAscore",
        author="Reymond Group/Molecular AI AstraZeneca",
        author_email="amol.thakkar@dcb.unibe.ch",
        license="MIT",
        license_file="LICENSE",
        zip_safe=False,
        install_requires=["scikit-learn", "xgboost", "h5py", "click", "tqdm"],
        extras_require={
            "retraining": [
                "matplotlib",
                "ModifiedNB",
                "numpy",
                "optuna",
                "pandas",
                "scipy",
                "seaborn",
                "swifter",
                "tables",
            ]
        },
        entry_points={
            "console_scripts": ["RAscore = RAscore.command_line_interface:main"]
        },
        include_package_data=True,
        python_requires=">=3.7",
    )
