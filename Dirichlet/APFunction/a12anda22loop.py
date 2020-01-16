from dolfin import *
import array 
hom12 = array.array('d', [])
hom22 = array.array('d', [])
error12 = array.array('d', [])
error22 = array.array('d', [])    

with open("a22.dat","w+") as dataf:
  for i in range(0, 30):

    R = 2*(i+1)
    mesh = RectangleMesh(Point(-R,-R),Point(R,R),60*(i+1)+200,60*(i+1)+200)
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
    #solve(a == L1, u1, bc, solver_parameters={"linear_solver":"mumps"})
    solve(a == L2, u2, bc, solver_parameters={"linear_solver":"mumps"})


    here12 = assemble(g1*grad(u2)[0]*dx(mesh))
    here12 = here12/(4*R*R)
    here22 = assemble(g2*dx(mesh))
    here22 = here22 + assemble(g2*grad(u2)[1]*dx(mesh))
    here22 = here22/(4*R*R)
    hom12.append(here12);
    hom22.append(here22);

    ref2 = Constant(7.000450820902004)
    ref1 = Constant(1.0140024932824005e-20)
    here2 = sqrt((hom12[i]-ref1)*(hom12[i]-ref1))
    error12.append(here2)
    here3 = sqrt((hom22[i]-ref2)*(hom22[i]-ref2))
    error22.append(here3)
    print(str(2*(i+1))+" "+str(hom12[i])+" "+str(error12[i])+" "+str(hom22[i])+" "+str(error22[i]))      
    dataf.write(str(2*i+2)+" "+str(hom12[i])+" "+str(error12[i])+" "+str(hom22[i])+" "+str(error22[i])+"\n")

  dataf.close()
