
# FerQ: Fermat Quotient Reformulation of High-Order Binary Optimization
*Accepted paper at NeurIPS26*

High-order binary optimisation (HOBO) problems, defined as the minimisation of functions involving interactions among three or more binary variables, are prevalent in machine learning and quantum computing. The minimisation of these high-order energy functions remains computationally intractable without quadratization or reformulation. This work introduces FerQ, a universal and auxiliary-free reformulation of high-order energy functions based on Fermat quotient expansion. For any degree-$d$ monomial over binary variables, FerQ provides an exact, closed-form representation as a rational linear combination of Fermat quotients evaluated on a scalar aggregate, with coefficients determined by a single matrix inversion. FerQ is evaluated against twelve existing energy function transformation methods on $k$-SAT and Max-$k$-SAT benchmarks from four databases, as well as on random $p$-spin glass and $k$-local Hamiltonian instances. FerQ achieves the highest satisfaction and weighted satisfaction rates across all benchmarks, while incurring the lowest CPU runtime. To enable deployment on quantum annealers, we derive FerQ-Bc, a qubit-efficient embedding scheme that maps the reformulated energies onto quadratic unconstrained binary optimisation (QUBO) form, achieving good ancilla efficiency in high-degree regimes.

# Code usage
```
reducers/.. # modular script for HOBO to QUBO methods
run/.. # executable scripts to reproduce results
```

# Contact
Phuong-Nam Nguyen
Graduate Research, School of Information Technology
Email: namnp.0510@gmail.com
