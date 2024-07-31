# lmp_reax
post code for paper "Assessment on the Rings Cleavage Mechanism of Polycyclic Aromatic Hydrocarbons in Supercritical Water: A ReaxFF Molecular Dynamics Study"
# Requirements
- python3.11
- python packdges: pandas, numpy, matplotlib, openbabel (can be installed from `pip3`)
# Installation
```bash
git clone https://github.com/haoxy97/lmp_reax.git
cd lmp_reax/src
pip3 install .
```
# Usage
open a terminal
`python3 -m "import lmp_reax;lmp_reax.run()"`
# example
```bash
cd lmp_reax/example
lmp -i user.inp # need run lammps first to generate traj file.
python3 -m "import lmp_reax;lmp_reax.run()
```
