#################################
PyMDP: A decomposition tool for multi-direction 3D printing  
#################################

Multi-directional 3D printing by robotic or automation systems has changed the way of traditional layer-wise printing. As a strong complementary of additive manufacturing, multi-directional printing has the capability of decreasing or eliminating the need for support structures.

----
Dependency
----

`Eigen <http://eigen.tuxfamily.org/>`_  `CGAL <https://www.cgal.org/>`_ `PyBind11 <http://github.com/pybind/pybind11/>`_


-------
Install
-------

We use CMake (>=3.16) and vcpkg to facilate the compilation process. You can download and install CMake from their official website, and install vcpkg by

.. code-block:: bash

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg integrate install

Next, you will need to install CGAL dependency:

.. code-block:: bash

    vcpkg install eigen
    vcpkg install cgal
    

Note: if you are running on a Windows system, vcpkg will install 32-bit package by default. In this case, you might need to use the following command

.. code-block:: bash

    vcpkg install eigen:x64-windows
    vcpkg install cgal:x64-windows

Then you can easily install the library by using the following command.

.. code-block:: bash

    pip install . --install-option="--vcpkg=YOUR_VCPKG_FOLDER"

Please change "YOUR_VCPKG_FOLDER" to the folder where VCPKG is installed.

-------
Demo
-------

.. code-block:: python

    from pymdp import BGS
    
    if __name__ == "__main__":
        proc = BGS(filename='kitten.off')
        proc.set_beam_width(10)
        proc.set_output_folder('kitten')
        proc.start_search()


-------
Credits
-------
We really appreciate if your scientific publications resulting from the projects that make use of PyMDP would cite our work.

.. code-block:: bibtex

    @article{wu2019general,
    title={General Support-Effective Decomposition for Multi-Directional 3-D Printing},
    author={Wu, Chenming and Dai, Chengkai and Fang, Guoxin and Liu, Yong-Jin and Wang, Charlie CL},
    journal={IEEE Transactions on Automation Science and Engineering},
    year={2019},
    publisher={IEEE}
    }

-------
License
-------
This library is ONLY for research purposes at your university (research institution). 
In no event shall the author be liable to any party for direct, indirect, special, incidental, or consequential damage arising out of the use of this program.