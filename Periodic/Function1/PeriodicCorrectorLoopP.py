from dolfin import *
import array 
hom = array.array('d', [])
error = array.array('d', [])
with open("PeriodicApproxFunction1.dat","w+") as dataf:

    exact = Constant(2.756777970694853);

    for i in range(0, 60):
            RS =  -2*i-2
            LS =  2*i+2
            mesh = RectangleMesh(Point(RS,RS), Point(LS, LS), 200+60*(i+1),200+60*(i+1))
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
                    else:
                        y[0] = x[0]
                        y[1] = x[1] - (LS-RS)
            
            W = FiniteElement("Lagrange",mesh.ufl_cell(),2)
            R = FiniteElement("R",mesh.ufl_cell(),0)
            TH = W * R
            V = FunctionSpace(mesh, TH, constrained_domain=PeriodicBoundary())

            # No Dirichlet Boundary Condtions
            bcs= []

            # Define variational problem
            (u,c) = TrialFunction(V)
            (v,d) = TestFunction(V)
            C = Expression("(2+1.8*sin(2*pi*x[0]))/(2+1.8*cos(2*pi*x[1]))+(2+sin(2*pi*x[1]))/(2+1.8*cos(2*pi*x[0]))",degree=2)
            a = C*inner(grad(u), grad(v))*dx + c*v*dx + u*d*dx
            L = -C*grad(v)[0]*dx


            # Compute solution
            u = Function(V)
            solve(a == L, u, bcs, solver_parameters={"linear_solver":"mumps"})
            (ss,c) = u.split()


            here1 = assemble(C*dx(mesh))
            here1 = here1 + assemble(C*grad(ss)[0]*dx(mesh))
            here1 = here1/((LS-RS)*(LS-RS))
            hom.append(here1)
            here2 = abs(here1-exact)
            error.append(here2)
      
            print(str(LS)+" "+str(hom[i])+" "+str(error[i]))
            dataf.write(str(LS)+" "+str(hom[i])+" "+str(error[i])+"\n")

    dataf.close()
