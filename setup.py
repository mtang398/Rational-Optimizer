import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


building_extension = any(
    command in sys.argv for command in ("build_ext", "bdist_wheel", "install")
)
if building_extension and CUDA_HOME is None:
    raise RuntimeError(
        "CUDA_HOME is required to build the fused GRAIN extension. "
        "Load CUDA 12.8 or set CUDA_HOME before building."
    )

extensions = []
commands = {}
if CUDA_HOME is not None:
    extensions = [
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
    ]
    commands = {"build_ext": BuildExtension}


setup(
    name="rational-opt",
    version="0.1.0",
    description="GRAIN activations and the TILLER optimizer for PyTorch",
    python_requires=">=3.12",
    package_dir={"rational_opt": "activation/rational_opt"},
    packages=[
        "rational_opt",
        "optimizer_design",
        "optimizer_design._tiller",
        "training",
        "experiments",
        "experiments.protocol",
    ],
    ext_modules=extensions,
    cmdclass=commands,
)
