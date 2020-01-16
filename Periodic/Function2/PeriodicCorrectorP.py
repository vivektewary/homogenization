#import matplotlib.pyplot as plt
from dolfin import *

RS = -25.5
LS = 25.5

mesh = RectangleMesh(Point(RS,RS), Point(LS, LS), 4420, 4420)

class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((abs(x[0]-RS)<DOLFIN_EPS or abs(x[1]-RS)<DOLFIN_EPS) and 
                (not ((near(x[0], RS) and near(x[1], LS)) or 
                        (near(x[0], LS) and near(x[1], RS)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], LS) and near(x[1], LS):
            y[0] = x[0] - 2*LS
            y[1] = x[1] - 2*LS
        elif near(x[0], LS):
            y[0] = x[0] - 2*LS
            y[1] = x[1]
        else :
            y[0] = x[0]
            y[1] = x[1] - 2*LS


W = FiniteElement("Lagrange",mesh.ufl_cell(),1)
R = FiniteElement("R",mesh.ufl_cell(),0)
TH = W * R
V = FunctionSpace(mesh, TH, constrained_domain=PeriodicBoundary())

# No Dirichlet Boundary Condtions
bcs= []

# Define variational problem
(u,c) = TrialFunction(V)
(v,d) = TestFunction(V)
C = Expression("1+30*(2+sin(2*DOLFIN_PI*x[0])*sin(2*DOLFIN_PI*x[1]))",degree=2)
a = C*inner(grad(u), grad(v))*dx + c*v*dx + u*d*dx
L = -C*grad(v)[0]*dx


# Compute solution
u = Function(V)
solve(a == L, u, bcs, solver_parameters={"linear_solver":"mumps"})
(ss,c) = u.split()


hom=assemble(C*dx(mesh));
hom = hom + assemble(C*grad(ss)[0]*dx(mesh))
hom = hom/((LS-RS)*(LS-RS))
print(hom)


# Plot solution
#plot(ss)
#plt.show()
