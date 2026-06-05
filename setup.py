from setuptools import setup, find_packages

setup(
    name="shopsense",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "sqlalchemy>=2.0",
        "psycopg2-binary",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "lightgbm",
        "shap>=0.44",
        "statsmodels",
        "mlflow>=2.0",
        "fastapi",
        "uvicorn",
        "pytest",
        "matplotlib",
        "seaborn",
        "plotly",
        "scipy"
    ],
)
