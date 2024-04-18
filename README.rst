===========================================================
PyMDP: A decomposition tool for multi-direction 3D printing  
===========================================================

.. image:: /img/figRepo.jpg

Multi-directional 3D printing by robotic arms or multi-axis systems is a new way of manufacturing. As a strong complementary of layer-wise additive manufacturing, multi-directional printing has the capability of decreasing or eliminating the need for support structures.

------
Notice
------
This library is **no longer actively maintained**. If you come across any complications during the compilation process, we suggest exploring the option of using a previous version of VCPKG from the year 2022. This approach has proven to be effective for numerous users who encountered similar issues and reached out to us via email.

----------
Dependency
----------

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

    vcpkg install eigen3
    vcpkg install cgal
    

Note: If you are using a Windows system, please be aware that vcpkg will install the 32-bit package as the default option. If you encounter this situation, you may need to utilize the following command.

.. code-block:: bash

    vcpkg install eigen3:x64-windows
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


We have recently introduced a learning-based approach to enhance the original search algorithm, utilizing learning-to-rank techniques. The source codes for this method can be found in the "learning_based.py" file, which is available for access.



-------
Credits
-------
We kindly request that any scientific publications utilizing PyMDP cite our work, as we greatly appreciate your support.

.. code-block:: bibtex
    
    @inproceedings{wu2017robofdm,
      title={RoboFDM: A robotic system for support-free fabrication using FDM},
      author={Wu, Chenming and Dai, Chengkai and Fang, Guoxin and Liu, Yong-Jin and Wang, Charlie CL},
      booktitle={2017 IEEE International Conference on Robotics and Automation (ICRA)},
      pages={1175--1180},
      year={2017},
      organization={IEEE}
    }

.. code-block:: bibtex

    @article{wu2019general,
    title={General Support-Effective Decomposition for Multi-Directional 3-D Printing},
    author={Wu, Chenming and Dai, Chengkai and Fang, Guoxin and Liu, Yong-Jin and Wang, Charlie CL},
    journal={IEEE Transactions on Automation Science and Engineering},
    year={2019},
    publisher={IEEE}
    }

.. code-block:: bibtex

    @article{wu2020learning,
      title={Learning to accelerate decomposition for multi-directional 3D printing},
      author={Wu, Chenming and Liu, Yong-Jin and Wang, Charlie CL},
      journal={IEEE Robotics and Automation Letters},
      volume={5},
      number={4},
      pages={5897--5904},
      year={2020},
      publisher={IEEE}
    }


In our learning-to-accelerate work, we use `urank <https://github.com/XiaofengZhu/uRank_uMart>`_  impelementation provided by Xiaofeng Zhu. Please consider cite their work if you also found it helpful.

-------
License
-------
This library is intended solely for research purposes within your university or research institution. The author shall not be held liable to any party for any direct, indirect, special, incidental, or consequential damages resulting from the utilization of this program.
