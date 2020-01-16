from dolfin import *
import array 
hom11 = array.array('d', [])
hom21 = array.array('d', [])
error11 = array.array('d', [])
error21 = array.array('d', [])    

with open("a11.dat","w+") as dataf:
  for i in range(0, 30):

    R = 2*i+1.5
    mesh = RectangleMesh(Point(-R,-R),Point(R,R),60*i+200,60*i+200)
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
    a = inner((C*grad(u)), grad(v))*dx
    L1 = -g1*grad(v)[0]*dx
    L2 = -g2*grad(v)[1]*dx

    # Compute solution
    u1 = Function(V)
    u2 = Function(V)
    solve(a == L1, u1, bc, solver_parameters={"linear_solver":"mumps"})
    #solve(a == L2, u2, bc, solver_parameters={"linear_solver":"mumps"})


    here11 = assemble(g1*dx(mesh))
    here11 = here11 + assemble(g1*grad(u1)[0]*dx(mesh))
    here11 = here11/(4*R*R)
    here21 = assemble(g2*grad(u1)[1]*dx(mesh))
    here21 = here21/(4*R*R)
    hom11.append(here11);
    hom21.append(here21);

    ref1 = Constant(3.9082266077013297)
    ref2 = Constant(1.7118724093553244e-19)
    here2 = sqrt((hom11[i]-ref1)*(hom11[i]-ref1))
    error11.append(here2)
    here3 = sqrt((hom21[i]-ref2)*(hom21[i]-ref2))
    error21.append(here3)
    print(str(2*i+1.5)+" "+str(hom11[i])+" "+str(error11[i])+" "+str(hom21[i])+" "+str(error21[i]))      
    dataf.write(str(2*i+1.5)+" "+str(hom11[i])+" "+str(error11[i])+" "+str(hom21[i])+" "+str(error21[i])+"\n")

  dataf.close()
