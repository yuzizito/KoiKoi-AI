import logging
logging.getLogger("torch.utils.cpp_extension").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", message=".*Error checking compiler version for cl.*")

import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

if sys.platform == 'win32':
    # WindowsのMSVC向け最適化フラグ
    cflags = ['/O2', '/utf-8', '/std:c++17', '/fp:fast', '/arch:AVX2', '/openmp:llvm']
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

# --- /showIncludes オプションをあらゆる引数から完全に抹消するカスタムビルダークラス ---
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        if getattr(self, 'compiler', None) and self.compiler.compiler_type == 'msvc':
            orig_compile = self.compiler.compile
            
            def patched_compile(*args, **kwargs):
                # 1. キーワード引数 (kwargs) 内のリストを徹底的にクリーニング
                for key in list(kwargs.keys()):
                    if isinstance(kwargs[key], list):
                        kwargs[key] = [x for x in kwargs[key] if x != '/showIncludes']
                
                # 2. 位置引数 (args) の中にリストがあればそこからも除去
                cleaned_args = []
                for arg in args:
                    if isinstance(arg, list):
                        cleaned_args.append([x for x in arg if x != '/showIncludes'])
                    else:
                        cleaned_args.append(arg)
                
                return orig_compile(*cleaned_args, **kwargs)
            
            self.compiler.compile = patched_compile
            
        super().build_extensions()

setup(
    name='koikoicore',
    version='1.0',
    description='KoiKoi Core Logic in C++ with LibTorch',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)