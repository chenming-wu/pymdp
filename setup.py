import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from distutils.version import LooseVersion

VCPKG = None

try:
    # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:
    # for pip <= 9.0.3
    from pip.req import parse_requirements



class InstallCommand(install):
    user_options = install.user_options + [
        ('vcpkg=', None, "<Directory to your vcpkg>"), # a 'flag' option
        #('someval=', None, None) # an option that takes a value
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.vcpkg = None
        #self.someval = None

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        global VCPKG
        VCPKG = self.vcpkg # will be 1 or None
        install.run(self)

def load_requirements(fname):
    reqs = parse_requirements(fname, session="test")
    return [str(ir.req) for ir in reqs]


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        cmake_version = LooseVersion(
            re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < LooseVersion('3.16.0'):
            raise RuntimeError("CMake >= 3.16.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        global VCPKG
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        build_type = os.environ.get("BUILD_TYPE", "Release")
        build_args = ['--config', build_type]

        # Pile all .so in one place and use $ORIGIN as RPATH
        cmake_args += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE"]
        cmake_args += ["-DCMAKE_INSTALL_RPATH={}".format("$ORIGIN")]
        
        if VCPKG != None:
            vcpkg_cmake = os.path.join(
                str(VCPKG), "scripts", "buildsystems", "vcpkg.cmake")
            cmake_args += ["-DCMAKE_TOOLCHAIN_FILE="+vcpkg_cmake]
                
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                build_type.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + build_type]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] +
                              cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake',
                               '--build', '.',
                               '--target', ext.name
                               ] + build_args,
                              cwd=self.build_temp)


setup(
    name='pymdp',
    version=0.1,
    author='Chenming Wu',
    author_email='wcm1994@gmail.com',
    description='A python package for multi-directional printing decomposition',
    long_description=open("README.rst").read(),
    ext_modules=[CMakeExtension('RoboFDM')],
    packages=find_packages(),
    cmdclass=dict(install=InstallCommand, build_ext=CMakeBuild),
    url="https://github.com/chenming-wu/pymdp",
    zip_safe=False,
    install_requires=load_requirements("requirements.txt"),
)
