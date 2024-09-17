# Educational Conduction-Convection Equation Solving Using Physics-Informed Neural Networks (PINN) and Finite Volume Method (FVM)

This is an educational repository for using PINN for solving steady-state conservation equation including both conduction (diffusion) and convection terms. The partial differential equation (PDE) governing such problem is

$$\vec{v} \cdot \nabla \theta = \nabla \cdot \left( \kappa \nabla \theta\right) &emsp; &emsp; &emsp; (1)$$

where $\vec{v}$ is the known velocity field, $\theta$ is the conserved dependent varaible depending on $\vec{x} = (x , y)$ and $\kappa$ is the diffusivity.

To examine performance of PINN in solving (1), the solution obtained from PINN is compared to the one from theory as well as the FVM. One possible solution to (1) for constant $\vec{u}$ and $\kappa$ is:

$$\theta = A + B \ exp\left(\frac{\vec{u} \cdot \vec{x}}{\kappa}\right) &emsp; &emsp; &emsp; (2)$$

in which $A$ and $B$ are constants. By applying (2) as a Dirichlet boundary condition, the solution of (1) inside a domain will be forced to follow (2) as well. 

Scope:
* Two-dimensional domain
* Known and constant velocity field, $\vec{u}$
* Constant diffusivity, $\kappa$

### Testing: 
pytest

### Requirements:
requirements.txt
  
