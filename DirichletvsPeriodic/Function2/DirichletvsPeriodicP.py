from dolfin import *
import array 
hom1 = array.array('d', [])
hom2 = array.array('d', [])
H1error = array.array('d', [])
homerror = array.array('d', [])

with open("DirichletvsPeriodicFunction2.dat","w+") as dataf:
  for i in range(0, 60):

    #Dirichlet Problem

    RS = -i-1.1
    LS = i+1.1

    mesh = RectangleMesh(Point(RS,RS), Point(LS, LS), 100+3*(i+1)*(i+1), 100+3*(i+1)*(i+1))
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define Dirichlet boundary (x = 0 or x = 1) 
    def boundary(x, on_boundary):
         return on_boundary

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, boundary)


    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    #C = Expression("(2+1.8*sin(2*pi*x[0]))/(2+1.8*cos(2*pi*x[1]))+(2+sin(2*pi*x[1]))/(2+1.8*cos(2*pi*x[0]))",degree=5)
    C = Expression("1+30*(2+sin(2*DOLFIN_PI*x[0])*sin(2*DOLFIN_PI*x[1]))",degree=2)
    f = Expression("x[0]",degree=2)
    a = C*inner((grad(u)), grad(v))*dx
    L = -C*grad(v)[0]*dx

    # Compute solution
    w1 = Function(V)
    solve(a == L, w1, bc, solver_parameters={"linear_solver":"mumps"})

    here1 = assemble(C*dx(mesh))
    here1 = here1 + assemble(C*grad(w1)[0]*dx(mesh))
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

    V = FiniteElement("Lagrange",mesh.ufl_cell(),1)
    R = FiniteElement("R",mesh.ufl_cell(),0)
    TH = V * R
    W = FunctionSpace(mesh, TH, constrained_domain=PeriodicBoundary())

    # No Dirichlet Boundary Condtions
    bcs= []

    # Define variational problem
    (u,c) = TrialFunction(W)
    (v,d) = TestFunction(W)
    #C = Expression("(2+1.8*sin(2*pi*x[0]))/(2+1.8*cos(2*pi*x[1]))+(2+sin(2*pi*x[1]))/(2+1.8*cos(2*pi*x[0]))",degree=2)
    C = Expression("1+30*(2+sin(2*DOLFIN_PI*x[0])*sin(2*DOLFIN_PI*x[1]))",degree=2)
    a = C*inner(grad(u), grad(v))*dx + c*v*dx + u*d*dx
    L = -C*grad(v)[0]*dx


    # Compute solution
    ss = Function(W)
    solve(a == L, ss, bcs, solver_parameters={"linear_solver":"mumps"})
    (w2,c) = ss.split()


    here2 = assemble(C*dx(mesh))
    here2 = here2 + assemble(C*grad(w2)[0]*dx(mesh))
    here2 = here2/((LS-RS)*(LS-RS))
    hom2.append(here2)
    here3 = abs(hom1[i]-hom2[i])
    homerror.append(here3)
    here4 = sqrt(assemble(inner(grad(w1-w2),grad(w1-w2))*dx(mesh)))
    here4 = here4/(LS-RS)
    H1error.append(here4)
    
    print(str(LS)+" "+str(hom1[i])+" "+str(hom2[i])+" "+str(homerror[i])+" "+str(H1error[i]))
    dataf.write(str(LS)+" "+str(hom1[i])+" "+str(hom2[i])+" "+str(homerror[i])+" "+str(H1error[i])+"\n")

  dataf.close()
