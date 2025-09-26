from setuptools import setup, Extension
import numpy

extensions = [
    Extension(
        "msanomalydetector._anomaly_kernel_cython",
        ["msanomalydetector/_anomaly_kernel_cython.c"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    name="msanomalydetector",
    ext_modules=extensions,
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)