import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

if sys.platform == 'win32':
    # WindowsのMSVC向け最適化フラグ
    cflags = ['/O2', '/utf-8', '/std:c++17', '/fp:fast', '/arch:AVX2']
else:
    # Linux/macOS向け最適化フラグ
    cflags = ['-O3', '-std=c++17', '-ffast-math', '-march=native']

ext_modules = [
    CppExtension(
        name='koikoicore',
        sources=['koikoicore.cpp'],
        extra_compile_args=cflags
    ),
]

setup(
    name='koikoicore',
    version='1.0',
    description='KoiKoi Core Logic in C++ with LibTorch',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)