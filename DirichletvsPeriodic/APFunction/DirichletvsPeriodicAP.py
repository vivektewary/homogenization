from dolfin import *
import array 
hom1 = array.array('d', [])
hom2 = array.array('d', [])
H1error = array.array('d', [])
homerror = array.array('d', [])

with open("DirichletvsPeriodicAPFunction.dat","w+") as dataf:
  for i in range(0, 30):

    #Dirichlet Problem

    RS = -2*i-1.5
    LS = 2*i+1.5

    mesh = RectangleMesh(Point(RS,RS), Point(LS, LS), 200+60*i, 200+60*i)
    V1 = FunctionSpace(mesh, "Lagrange", 2)

    # Define Dirichlet boundary (x = 0 or x = 1) 
    def boundary(x, on_boundary):
         return on_boundary

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V1, u0, boundary)


    # Define variational problem
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    pi = Constant(DOLFIN_PI)
    zero = Constant(0.0)
    g1 = Expression("4+cos(2*pi*(x[0]+x[1]))+cos(2*pi*sqrt(2)*(x[0]+x[1]))",degree=2)
    g2 = Expression("6+(sin(2*pi*x[0]))*(sin(2*pi*x[0]))+(sin(2*pi*sqrt(2)*x[0]))*(sin(2*pi*sqrt(2)*x[0]))",degree=2)
    C = as_matrix(((g1,zero),
                   (zero,g2)))
    a1 = inner((C*grad(u1)), grad(v1))*dx
    L1 = -g1*grad(v1)[0]*dx

    # Compute solution
    w1 = Function(V1)
    solve(a1 == L1, w1, bc, solver_parameters={"linear_solver":"mumps"})

    here1 = assemble(g1*dx(mesh))
    here1 = here1 + assemble(g1*grad(w1)[0]*dx(mesh))
    here1 = here1/((LS-RS)*(LS-RS))
    hom1.append(here1)

    #Periodic Problem

    class PeriodicBoundary(SubDomain):
      # Left boundary is "target domain" G
      def inside(self, x, on_boundary):
      # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((abs(x[0]-RS)<DOLFIN_EPS or abs(x[1]-RS)<DOLFIN_EPS) and 
          (not ((near(x[0], RS) and near(x[1], LS)) or 
            (near(x[0], LS) and near(x[1], RS)))) and on_boundary)

      def map(self, x, y):
        if near(x[0], LS) and near(x[1], LS):
          y[0] = x[0] - (LS-RS)
          y[1] = x[1] - (LS-RS)
        elif near(x[0], LS):
          y[0] = x[0] - (LS-RS)
          y[1] = x[1]
        else :
          y[0] = x[0]
          y[1] = x[1] - (LS-RS)


    W = FiniteElement("Lagrange",mesh.ufl_cell(),2)
    R = FiniteElement("R",mesh.ufl_cell(),0)
    TH = W * R
    V2 = FunctionSpace(mesh, TH, constrained_domain=PeriodicBoundary())

    # No Dirichlet Boundary Condtions
    bcs= []
    # Define variational problem
    (u2,c) = TrialFunction(V2)
    (v2,d) = TestFunction(V2)
    pi = Constant(DOLFIN_PI)
    zero = Constant(0.0)
    g1 = Expression("4+cos(2*pi*(x[0]+x[1]))+cos(2*pi*sqrt(2)*(x[0]+x[1]))",degree=2)
    g2 = Expression("6+(sin(2*pi*x[0]))*(sin(2*pi*x[0]))+(sin(2*pi*sqrt(2)*x[0]))*(sin(2*pi*sqrt(2)*x[0]))",degree=2)
    C = as_matrix(((g1,zero),
                   (zero,g2)))
    a2 = inner((C*grad(u2)), grad(v2))*dx + c*v2*dx + u2*d*dx
    L2 = -g1*grad(v2)[0]*dx

    # Compute solution
    ss = Function(V2)
    solve(a2 == L2, ss, bcs, solver_parameters={"linear_solver":"mumps"})
    (w2,c) = ss.split()


    here2 = assemble(g1*dx(mesh))
    here2 = here2 + assemble(g1*grad(w2)[0]*dx(mesh))
    here2 = here2/((LS-RS)*(LS-RS))
    hom2.append(here2)
    here3 = abs(hom1[i]-hom2[i])
    homerror.append(here3)
    here4 = sqrt(assemble(inner(grad(w1-w2),grad(w1-w2))*dx(mesh)))
    here4 = here4/(2*LS)
    H1error.append(here4)
    
    print(str(LS)+" "+str(hom1[i])+" "+str(hom2[i])+" "+str(homerror[i])+" "+str(H1error[i]))
    dataf.write(str(LS)+" "+str(hom1[i])+" "+str(hom2[i])+" "+str(homerror[i])+" "+str(H1error[i])+"\n")

  dataf.close()

for i in range(0, 30):
  print(str(2*i+1.5)+" "+str(hom1[i])+" "+str(hom2[i])+" "+str(homerror[i])+" "+str(H1error[i]))


