#!/usr/bin/env python3
import ctypes
import sys

libcudart = None
for libname in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so.10.2"):
    try:
        libcudart = ctypes.CDLL(libname)
        break
    except OSError:
        libcudart = None

if libcudart is None:
    print("libcudart nicht gefunden. Stelle sicher, dass die CUDA Runtime installiert ist.")
    sys.exit(1)

# cudaGetDeviceCount
cudaGetDeviceCount = libcudart.cudaGetDeviceCount
cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
cudaGetDeviceCount.restype = ctypes.c_int

cnt = ctypes.c_int()
res = cudaGetDeviceCount(ctypes.byref(cnt))
if res != 0:
    print(f"cudaGetDeviceCount returned error code {res}")
    sys.exit(1)

if cnt.value <= 0:
    print("Keine CUDA-Geräte gefunden.")
    sys.exit(1)

# cudaSetDevice(0) to initialize runtime on device 0
cudaSetDevice = libcudart.cudaSetDevice
cudaSetDevice.argtypes = [ctypes.c_int]
cudaSetDevice.restype = ctypes.c_int
res = cudaSetDevice(0)
if res != 0:
    print(f"cudaSetDevice returned error code {res}")
    # continue anyway

# cudaMemGetInfo
cudaMemGetInfo = libcudart.cudaMemGetInfo
cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
cudaMemGetInfo.restype = ctypes.c_int
free = ctypes.c_size_t()
total = ctypes.c_size_t()
res = cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
if res != 0:
    print(f"cudaMemGetInfo returned error code {res}")
    sys.exit(1)

GB = 1024 ** 3
print(f"GPU total: {total.value} bytes ({total.value/GB:.2f} GB)")
print(f"GPU free : {free.value} bytes ({free.value/GB:.2f} GB)")

suggested_gb = free.value / GB * 0.8
print(f"Vorschlag: GPU_MEM_LIMIT_GB = {suggested_gb:.2f} (ca. 80% des freien Speichers)")
