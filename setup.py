import setuptools

setuptools.setup(
    author="Giulio Cesare Mastrocinque Santo",
    name='HDSIdent',
    author_email="giuliosanto@gmail.com",
    version='0.0.2.1',
    description='HDSIdent is an open-source Python package that can be used to obtain historical data segments suitable for performing System Identification',
    license="MIT",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=["scikit-learn","scipy",
                      "pandas","numpy","matplotlib",
                      "seaborn","joblib","sympy"],
    zip_safe=True
)