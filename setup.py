import sys
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'koikoicore', # Pythonからimportする際の名前
        ['koikoicore.cpp'], # コンパイルするC++ファイル
        include_dirs=[pybind11.get_include()],
        language='c++',
        # WindowsのMSVCコンパイラ向け最適化フラグ
        extra_compile_args = ['/O2', '/utf-8', '/std:c++17']
    ),
]

setup(
    name='koikoicore',
    version='1.0',
    description='KoiKoi Core Logic in C++',
    ext_modules=ext_modules,
)