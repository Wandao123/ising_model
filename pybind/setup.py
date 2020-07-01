#!/usr/bin/env python

from setuptools import setup, Extension#, Command

#class BuildCommand(Command):
#    description = 'Building options'
#    user_options = [
#        ('vcpkg-root=', None, 'Path to vcpkg root.'),
#    ]
#
#    def initialize_options(self):
#        self.vcpkgRoot = None
#
#    def finalize_options(self):
#        pass
#
#    def run(self):
#        build_all_the_things()

setup(
    name='simulatorWithCpp',
    version='1.0',
    description='An Ising model simulator',
    author='Wandao123',
    author_email='20665675+Wandao123@users.noreply.github.com',
    url='https://github.com/Wandao123/ising_model',
    ext_modules=[Extension(
        'simulatorWithCpp', ['wrapper.cpp', '../cpp/simulator.cpp'],
        #include_dirs=['%VCPKG_ROOT%/installed/x64-windows/include'],
        extra_compile_args=['/std:c++17'],
        extra_link_args=['/MACHINE:X64'])],
    #cmdclass={ 'build': BuildCommand, },
)
