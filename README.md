The file QAOA_SVP.py provides the general framework for simulating the SVP Hamiltonian using the standard QAOA ansatz with the SWAP gate algorithm for a 1D geometry.
The file is preset to run a second-order QAOA.
This file can be used for any qubit count, and is preset for a 6 qubit system formulated through a 3D lattice and 2D qudits.

The file Modified_QAOA_12q.py is an extension of the general framework to include the determined driver Hamiltonian coefficients.
This file is written specifically for 12 qubit systems, and is preset for a 4D lattice and 3D qudits.

Both files are written to run immediately upon download for example lattices. 
These lattice basis vectors can be manually altered within the 'Gram' function, however this may require additional adjustment of the lat_dim and qudit_dim parameters.
