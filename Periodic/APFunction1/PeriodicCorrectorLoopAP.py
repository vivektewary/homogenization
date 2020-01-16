from dolfin import *
import array 
hom11 = array.array('d', [])
error11 = array.array('d', [])
hom21 = array.array('d', [])
error21 = array.array('d', [])
with open("PeriodicApproxAPFunction.dat","w+") as dataf:

        for i in range(0, 30):

                RS = -2*i-1.1
                LS = 2*i+1.1

                mesh = RectangleMesh(Point(RS,RS), Point(LS, LS), 100+80*i, 100+80*i)

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
                V = FunctionSpace(mesh, TH, constrained_domain=PeriodicBoundary())

                # No Dirichlet Boundary Condtions
                bcs= []

                # Define variational problem
                (u,c) = TrialFunction(V)
                (v,d) = TestFunction(V)
                pi = Constant(DOLFIN_PI)
                zero = Constant(0.0)
                g1 = Expression("4+cos(2*pi*(x[0]+x[1]))+cos(2*pi*sqrt(2)*(x[0]+x[1]))",degree=2)
                g2 = Expression("4+cos(2*pi*(x[0]-x[1]))+cos(2*pi*sqrt(2)*(x[0]-x[1]))",degree=2)
                C = as_matrix(((g1,zero),
                               (zero,g2)))
                a = inner(C*(grad(u)), grad(v))*dx + c*v*dx + u*d*dx
                L = -g1*grad(v)[0]*dx


                # Compute solution
                u = Function(V)
                solve(a == L, u, bcs, solver_parameters={"linear_solver":"mumps"})
                (ss,c) = u.split()


                here1 = assemble(g1*dx(mesh))
                here1 = here1 + assemble(g1*grad(ss)[0]*dx(mesh))
                here1 = here1/((LS-RS)*(LS-RS))
                hom11.append(here1)
                ref11 = Constant(3.8639808592068055)
                here2 = abs((hom11[i]-ref11))
                error11.append(here2)
                here3 = assemble(g1*grad(ss)[1]*dx(mesh))
                here3 = here3/((LS-RS)*(LS-RS))
                hom21.append(here3)
                ref21 = Constant(-0.1360213730552993)
                here4 = abs((hom21[i]-ref21))
                error21.append(here4)
                print(str(LS)+" "+str(hom11[i])+" "+str(error11[i])+" "+str(hom21[i])+" "+str(error21[i]))
                dataf.write(str(2*i+1.5)+" "+str(hom11[i])+" "+str(error11[i])+" "+str(hom21[i])+" "+str(error21[i])+"\n")

        dataf.close()
