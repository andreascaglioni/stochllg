Introduction
============
Approximation of the sample paths of the *Stochastic Landau-Lifshitz-Gilbert equation* from dynamic micromagnetics.

Installation & Usage
--------------------
For the moment, only the repository is available. In the future a package will
be released on PyPI.

#. Clone the repository:

   .. code-block:: console
    
    git clone <git@github.com:andreascaglioni/stochastic-llg-equation.git>
    
#. Install the dependencies:

   * The finite elements computations are run with dolfin-x, the Python interface to Fenics-x. See their website `fenicsproject.org <https://fenicsproject.org/>`_ for installation instructions (Conda environment *recommended*).
   
   * Running the code requires several basic Python dependencies listed in the `requirements.txt` file. You can install them with pip:
   
    .. code-block:: console
    
      pip install -r requirements.txt
    
#. Optionally, you can install ParaView to visualize the solutions from the examples stored in .xdmf files. Find more instructions on their website: `paraview.org <https://www.paraview.org/>`_.

Usage
-----
Run the examples from the root directory of the project with:
  
.. code-block:: python
  
  python3 examples/example_name.py

Features
--------
Here are the main features of the code:

- Implementation of the Tangent Plane Scheme (TPS) as in the publication :cite:`Akrivis2021Higher`. Here, the tangent plane constraint is understood in a "weak" :math:`L^2` sense and both the BDF time stepping and finite elements space discretizations are high-order.
- Lévy-Ciesielski parametrization of the Brownian motion: Given a vector of i.i.d. samples of the standard normal distribution, generate a sample path in time.
- Structured, modular implementation of the Tangent Plane Scheme, split into sub-functions as well as several supporting functions such as:

  - Computation of the error.
  - Computation of the inf-sup constant.
  - Exporting solutions to .XDMF files.

- Examples and documentation: Several examples illustrate the use of the library, including good practices e.g. defining the data in a separate file, commenting the code. Functions, especially in `src/`, are thoroughly documented (Google style).

Futures to be added in the future
----------------------------------

* Unit testing with Pytest
* High order version of the TPS
* More examples (external magnetic field, hysteresis, Skyrmions, ...)
* Automated Testing with *GitHub Actions*

Contributing
------------
Contributions are welcome! Please get in touch (see `Contact <#contact>`_).

License
-------
Distributed under the MIT License. See `LICENSE` for more information.

Contact
-------
If you have any questions, suggestions or comments, please get in touch with me:
Andrea Scaglioni `andreascaglioni.net/contacts <https://andreascaglioni.net/contacts/>`_
