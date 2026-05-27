from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="rational-opt",
    version="0.1.0",
    description="CUDA Rational 5/4 activation for PyTorch",
    package_dir={"": "activation"},
    packages=find_packages("activation"),
    ext_modules=[
        CUDAExtension(
            name="rational_opt._C",
            sources=[
                "activation/csrc/rational_ext.cpp",
                "activation/csrc/rational_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-DNDEBUG"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-DNDEBUG",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
