from setuptools import setup, find_packages

setup(
    name='trading_analysis',
    version='0.1.0',
    description='Paquete científico para análisis de trading basado en estadística y simulación',
    author='Núria Navarro',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'ipywidgets',
        'python-dotenv'
    ],
    include_package_data=True,
    zip_safe=False
)
