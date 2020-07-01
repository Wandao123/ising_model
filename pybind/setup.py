#!/usr/bin/env python

from setuptools import setup, Extension

setup(
    name='simulatorWithCpp',
    version='1.1',
    description='An Ising model simulator',
    author='Wandao123',
    author_email='20665675+Wandao123@users.noreply.github.com',
    url='https://github.com/Wandao123/ising_model',
    ext_modules=[Extension(
        'simulatorWithCpp', ['wrapper.cpp', '../cpp/simulator.cpp'],
        #include_dirs=['%VCPKG_ROOT%/installed/x64-windows/include'],
        extra_compile_args=['/std:c++17'],
        #extra_link_args=['/MACHINE:X64']
    )],
)
