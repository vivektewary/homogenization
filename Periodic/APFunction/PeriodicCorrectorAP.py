import matplotlib.pyplot as plt
from dolfin import *

RS = -20.5
LS = 20.5

mesh = RectangleMesh(Point(RS,RS), Point(LS, LS), 600, 600)

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
pi = Constant(DOLFIN_PI)
zero = Constant(0.0)
g1 = Expression("4+cos(2*pi*(x[0]+x[1]))+cos(2*pi*sqrt(2)*(x[0]+x[1]))",degree=2)
g2 = Expression("4+cos(2*pi*(x[0]+x[1]))+cos(2*pi*sqrt(2)*(x[0]+x[1]))",degree=2)
C = as_matrix(((g1,zero),
       (zero,g2)))
a = inner((C*grad(u)), grad(v))*dx
L = -g1*grad(v)[0]*dx


# Compute solution
u = Function(V)
solve(a == L, u, bcs)
(ss,c) = u.split()


hom=assemble(g1*dx(mesh));
hom = hom + assemble(g1*grad(ss)[0]*dx(mesh))
hom = hom/((LS-RS)*(LS-RS))
print(hom)


# Plot solution
plot(ss)
plt.show()
