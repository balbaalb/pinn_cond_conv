# Educational Conduction-Convection Equation Solving Using Physics-Informed Neural Networks (PINN) and Finite Volume Method (FVM)

This is an educational repository for using PINN for solving steady-state conservation equation including both conduction (diffusion) and convection terms. The partial differential equation (PDE) governing such problem is

$\begin{equation}
\vec{v} \cdot \nabla \theta = \nabla \left( \kappa \nabla \theta\right)
\end{equation}$

where $\vec{v}$ is the known velocity field, $\theta$ is the conserved dependent varaible depending on $\vec{x} = (x , y)$ and $\kappa$ is the diffusivity.

To examine peroformance of PINN in solving (1), the solution obtained from PINN is compared to the one from theory as well as the FVM. One possible solution to (1) for constant $\vec{u}$ and $\kappa$ is:

$\begin{equation}
\theta = A + B \: exp\left( 
    \frac{\vec{u} \cdot \vec{x}}{\kappa}
\right)
\end{equation}$

in which $A$ and $B$ are constants. By applying (2) as the Dirichlet boundary conditions, the solution of (1) inside a domain will be forced to follow (2) as well. 

Scope:
* Two-dimensional domain
* Known and constant velocity field, $\vec{u}$
* Constant diffusivity, $\kappa$

### Testing: 
pytest

### Requirements:
requirements.txt
  