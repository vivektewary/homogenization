from fenics import *

list_linear_solver_methods()

mesh = UnitSquareMesh(32,32)
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
C = Expression("(2+1.8*sin(2*pi*x[0]))/(2+1.8*cos(2*pi*x[1]))+(2+sin(2*pi*x[1]))/(2+1.8*cos(2*pi*x[0]))",degree=2)
a = C*inner((grad(u)), grad(v))*dx
L = -C*grad(v)[0]*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc,solver_parameters={"linear_solver":"superlu"})

hom=assemble(C*dx(mesh));
hom = hom + assemble(C*grad(u)[0]*dx(mesh))
print(hom)
