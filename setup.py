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
        extra_compile_args=['/O2'] if sys.platform == 'win32' else ['-O3']
    ),
]

setup(
    name='koikoicore',
    version='1.0',
    description='KoiKoi Core Logic in C++',
    ext_modules=ext_modules,
)