# B-subdifferential-variational-inequalities

This is the python proof-of-concept implementation of our accompanying article

> Y. Song, P.I. Barton, New Generalized Derivatives for Solving Variational Inequalities Using the Nonsmooth Newton Methods, in review.

In this work, we propose a new method for furnishing B-subdifferential elements of the normal map associated with a variation inequality (VI) problem. Then, the normal map equation system can be solved using variants of the nonsmooth Newton methods. This work was supported by the OCP Group, Morroco.

## Implementation content

This section outlines the various scripts in the [src](src) folder.

### [main_script.py](src/main_script.py)
An implementation of our new B-subdifferential-based VI solution method, which accepts VIs considered in the article.  For a given starting point, it solves the VI's normal map equation system using either the classic Newton method or the LP Newton method. A user must provide the Jacobians and Hessians of functions in the VI as predefined functions; see [exp1_input.py](src/exp1_input.py) for an example. Linear equality constraints must be reformulated as an inequality pair, or the implementation can be easily adapted to accept linear equalities. 

To use this implementation to reproduce Examples 6.1and 6.2, simply import the corresponding example data from the top and choose the desired Newton method at the bottom.



### [exp1_input.py](src/exp1_input.py)
The VI formulation in Example 6.1.

### [exp1_starting_point_NM.json](src/exp1_starting_point_NM.json)
The starting point for solving the normal map equation system in Example~6.1.

### [exp2_input.py](src/exp2_input.py)
The VI formulation in Example 6.2.

### [exp2_starting_point_NM.json](src/exp2_starting_point_NM.json)
A list of the starting points for solving the normal map equation system in Example~6.2.

### [exp2_starting_point_VI.json](src/exp2_starting_point_VI.json)
A list of the starting points for solving the VI's KKT system using the PATH solver in Example~6.2.

### [exp2_by_PATH.py](src/exp2_by_PATH.py)
The code for solving the VI's KKT system using the PATH solver in Example~6.2.

## References

- S.P. Dirkse, M.C. Ferris. The PATH solver: a nonmonotone stabilization scheme for mixed complementarity problems. *Optimization Methods and Software*, **5**: 123-156 (1995)
- F. Facchinei, A. Fischer, M. Herrich. An LP-Newton method: nonsmooth equations, KKT systems, and nonisolated solutions. *Mathematical Programming*, **146**: 1-36 (2014)
- M. Kojima, S. Shindo. Extension of Newton and quasi-Newton methods to systems of PC<sup>1</sup> equations. *Journal of the Operations Research Society of Japan*, **29**: 352-375 (1986)
