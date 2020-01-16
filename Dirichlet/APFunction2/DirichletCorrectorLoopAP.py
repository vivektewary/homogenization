from dolfin import *
import array 
hom11 = array.array('d', [])
hom12 = array.array('d', [])
hom21 = array.array('d', [])
hom22 = array.array('d', [])
error = array.array('d', [])    

with open("DirichletApproxAPFunctionii.dat","w+") as dataf:
  for i in range(0, 15):

    R = 2*i+1.5
    mesh = RectangleMesh(Point(-R,-R),Point(R,R),30*i*i+200,30*i*i+200)
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
    g1 = Expression("40+sin(2*pi*(x[0]))+cos(pi*sqrt(2)*(x[1]))",degree=2)
    g2 = Expression("30+sin(sqrt(3)*pi*x[0])+cos(pi*x[1])",degree=2)
    C = as_matrix(((g1,zero),
         (zero,g2)))
    a = inner((C*grad(u)), grad(v))*dx
    L1 = -g1*grad(v)[0]*dx
    L2 = -g2*grad(v)[1]*dx

    # Compute solution
    u1 = Function(V)
    u2 = Function(V)
    solve(a == L1, u1, bc, solver_parameters={"linear_solver":"mumps"})
    solve(a == L2, u2, bc, solver_parameters={"linear_solver":"mumps"})


    here11 = assemble(g1*dx(mesh))
    here11 = here11 + assemble(g1*grad(u1)[0]*dx(mesh))
    here11 = here11/(4*R*R)
    hom11.append(here11);

    here12 = Constant(0.0)
    here12 = here12 + assemble(g1*grad(u2)[1]*dx(mesh))
    here12 = here12/(4*R*R)
    hom12.append(here12);

    here21 = Constant(0.0)
    here21 = here21 + assemble(g2*grad(u1)[0]*dx(mesh))
    here21 = here21/(4*R*R)
    hom21.append(here21);

    here22 = assemble(g2*dx(mesh))
    here22 = here22 + assemble(g2*grad(u2)[1]*dx(mesh))
    here22 = here22/(4*R*R)
    hom22.append(here22);

    print(str(R)+" "+str(hom11[i])+" "+str(hom12[i])+" "+str(hom21[i])+" "+str(hom22[i]))

  for i in range(0, 15):
    here2 = sqrt((hom11[i]-here11)*(hom11[i]-here11)+(hom12[i]-here12)*(hom12[i]-here12)+(hom21[i]-here21)*(hom21[i]-here21)+(hom22[i]-here22)*(hom22[i]-here22))
    error.append(here2)
    dataf.write(str(2*i+1.5)+" "+str(hom11[i])+" "+str(hom12[i])+" "+str(hom21[i])+" "+str(hom22[i])+" "+str(error[i])+"\n")
  dataf.close()
