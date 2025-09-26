# stochllg (Tangent Plane Scheme for the Stochastic Landau-Lifshitz-Gilbert Equation)

![Image](https://github.com/user-attachments/assets/2beba6ae-fa3b-482e-86e9-ba09b71bd37b)

Approximation of the sample paths of the Stochastic Landau-Lifshitz-Gilbert equation from dynamic micromagnetics.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation
1. Clone the repo:
   ```sh
   git clone <git@github.com:andreascaglioni/stochastic-llg-equation.git>
   ```
2. Install dependencies:
   - The finite elements computations are run with dolfin-x, the Python interface to Fenics-x. See their website for installation instructions (Conda environment *recommended*):
   [https://fenicsproject.org/download/](https://fenicsproject.org/download/) 
   
   - Running the code requires several basic Python dependencies:
      ```sh
         pip install -r requirements.txt
      ```
      
3. Optionally, you can install ParaView to visualize the solutions from the examples stored in .xdmf files. Find more instructions on their website:
   [https://www.paraview.org/](https://www.paraview.org/)

## Usage
Run the examples from the root directory of the project with
```sh
python3 examples/example_<name>.py
```

Read the documentation at: [https://andreascaglioni.github.io/stochllg/](https://andreascaglioni.github.io/stochllg/)

## Features
- Implementation of the Tangent Plane Scheme (TPS) as in the publication:

   Akrivis G., Feischl M., Kovács B. and Lubich C.; (2021).  Higher-order linearly implicit full discretization of the Landau–Lifshitz–Gilbert equation. *Math. Comp.*, *90*, 995-1038. [DOI: 10.1090/mcom/3597](https://doi.org/10.1090/mcom/3597)

   In particular, the tangent plane constraint is implemented in a ``weak'' L2 sense and
both the BDF time stepping and finite elements space discretizations are high-order.

- Lévy-Ciesielski parametrization of the Brownian motion: Given n i.i.d. samples of the standard normal distribution, generate a sample path in time.

- Structured, modular implementation of the Tangent Plane Scheme, split into sub-functions as well as several supporting functions such as 
computation of the error, 
computation of the inf-sup constant, 
exporting solutions to .XDMF files.

- Examples and documentation: Several examples illustrate the use of the library, including good practices e.g. defining the data a separate file, commenting the code. Functions, especially in src/, are thoroughly documented (Google style).

- Future additions: 
   - [ ] Unit testing with Pytest, 
   - [ ] Documentation, 
   - [ ] High order version of the TPS,
   - [ ] More examples (external magnetic field, hysteresis, Skyrmions, ...),
   - [ ] Autormated Testing with *GitHub Actions*
   <!-- - [ ] Plot the initial condition m_0, the noise space correlation g, the effective field H_eff as .xdmf files -->

## Contributing
Contributions are welcome! Please get in touch (see [Contact](#contact)).

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
If you have any questions, suggestions or comments, please get in touch with me:
Andrea Scaglioni [andreascaglioni.net/contacts](https://andreascaglioni.net/contacts/)
