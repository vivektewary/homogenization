from fenics import *

mesh = RectangleMesh(Point(-60,-60),Point(60,60),5000,5000)
V = FunctionSpace(mesh, "Lagrange", 2)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x, on_boundary):
    return on_boundary

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
pi = Constant(DOLFIN_PI)
zero = Constant(0.0)
g1 = Expression("4+cos(2*pi*(x[0]+x[1]))+cos(2*pi*sqrt(2)*(x[0]+x[1]))",degree=2)
g2 = Expression("6+(sin(2*pi*x[0]))*(sin(2*pi*x[0]))+(sin(2*pi*sqrt(2)*x[0]))*(sin(2*pi*sqrt(2)*x[0]))",degree=2)
C = as_matrix(((g1,zero),
       (zero,g2)))
a = inner((C*grad(u)), grad(v))*dx+(1/5000)*u*v*dx
L = -g1*grad(v)[0]*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc,solver_parameters={"linear_solver":"mumps"})

hom=assemble(g1*dx(mesh));
hom = hom + assemble(g1*grad(u)[0]*dx(mesh))
print(hom)

# Plot solution
import matplotlib.pyplot as plt
plot(u)
plt.show()
