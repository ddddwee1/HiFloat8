from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hif8_quant',
    ext_modules=[
        CUDAExtension('hif8_quant', [
            'hif8_quant.cpp',
            'hif8_quant_cuda.cu',
        ],
        #extra_compile_args=['-std=c++17'], 
        extra_link_args=['-lgomp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
