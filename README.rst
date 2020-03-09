#################################
PyMDP: A decomposition tool for multi-direction 3D printing  
#################################

Multi-directional 3D printing by robotic or automation systems has changed the way of traditional layer-wise printing. As a strong complementary of additive manufacturing, multi-directional printing has the capability of decreasing or eliminating the need for support structures.

----
Dependency
----
CGAL: https://www.cgal.org

Boost: https://www.boost.org

PyBind11: http://github.com/pybind/pybind11/

-------
Install
-------

Make sure you have installed CMake (>=3.16), simply use the following command to install the library.

.. code-block:: bash

    python -m pip install pymdp

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