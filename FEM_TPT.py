# This code is a python adaptation of the code in the paper:
# Title: Remarks around 50 lines of Matlab: short finite element implementation
# Authors: Jochen Alberty, Carsten Carstensen and Stefan A. Funken
# Journal: Numerical Algorithms 20 (1999) 117–137

# Solves the committor problem using the FEM
# grad * (exp(-beta V(x,y) grad q(x,y)) = 0, q(bdry A) = 0, q(bdry B) = 1
# the potential used is the Face potential defined in this cell

from matplotlib.cbook import pts_to_midstep
import numpy as np
import math
import matplotlib.pyplot as plt
import csv 
import scipy
import scipy.linalg
from scipy.sparse import csr_matrix


def put_pts_on_circle(xc,yc,r,n):
    t = np.linspace(0,math.pi*2,n+1)
    pts = np.zeros((n,2))
    pts[:,0] = xc+r*np.cos(t[0:n])
    pts[:,1] = yc+r*np.sin(t[0:n])
    return pts

def reparametrization(path,h):
    dp = path - np.roll(path,1,axis = 0);
    dp[0,:] = 0;
    dl = np.sqrt(np.sum(dp**2,axis=1));
    lp = np.cumsum(dl);
    len = lp[-1];
    lp = lp/len; # normalize
    npath = int(round(len/h));
    g1 = np.linspace(0,1,npath)
    path_x = np.interp(g1,lp,path[:,0])
    path_y = np.interp(g1,lp,path[:,1])
    path = np.zeros((npath,2))
    path[:,0] = path_x
    path[:,1] = path_y
    return path

def find_ABbdry_pts(pts,xc,yc,r,h0):
    ind = np.argwhere(np.sqrt((pts[:,0]-xc)**2+(pts[:,1]-yc)**2)-r < h0*1e-2)
    Nind = np.size(ind)
    ind = np.reshape(ind,(Nind,))
    return Nind,ind

def stima3(verts):
    Aux = np.ones((3,3))
    Aux[1:3,:] = np.transpose(verts)
    rhs = np.zeros((3,2))
    rhs[1,0] = 1
    rhs[2,1] = 1
    G = np.zeros((3,3))
    G[:,0] = np.linalg.solve(Aux,rhs[:,0])
    G[:,1] = np.linalg.solve(Aux,rhs[:,1])
    M = 0.5*np.linalg.det(Aux)*np.matmul(G,np.transpose(G))
    return M

def FEM_committor_solver(pts,tri,Aind,Bind,fpot,beta):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    Dir_bdry = np.hstack((Aind,Bind))
    free_nodes = np.setdiff1d(np.arange(0,Npts,1),Dir_bdry,assume_unique=True)

    A = csr_matrix((Npts,Npts), dtype = np.float).toarray()
    b = np.zeros((Npts,1))
    q = np.zeros((Npts,1))
    q[Bind] = 1

    # stiffness matrix
    for j in range(Ntri):
        v = pts[tri[j,:],:] # vertices of mesh triangle
        vmid = np.reshape(np.sum(v,axis=0)/3,(1,2)) # midpoint of mesh triangle
        fac = np.exp(-beta*fpot(vmid))
        ind = tri[j,:]
        indt = np.array(ind)[:,None]
        B = csr_matrix((3,3),dtype = np.float).toarray()
        A[indt,ind] = A[indt,ind] + stima3(v)*fac

    # load vector
    b = b - np.matmul(A,q)

    # solve for committor
    free_nodes_t = np.array(free_nodes)[:,None]
    q[free_nodes] = scipy.linalg.solve(A[free_nodes_t,free_nodes],b[free_nodes])
    q = np.reshape(q,(Npts,))
    return q

def reactive_current_and_transition_rate(pts,tri,fpot,beta,q):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    # find the reactive current and the transition rate
    Rcurrent = np.zeros((Ntri,2)) # reactive current at the centers of mesh triangles
    Rrate = 0
    Z = 0
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        qtri = q[ind]
        a = np.array([[verts[1,0]-verts[0,0],verts[1,1]-verts[0,1]],[verts[2,0]-verts[0,0],verts[2,1]-verts[0,1]]])
        b = np.array([qtri[1]-qtri[0],qtri[2]-qtri[0]])
        g = np.linalg.solve(a,b)
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        vmid = np.reshape(np.sum(verts,axis=0)/3,(1,2)) # midpoint of mesh triangle
        mu = np.exp(-beta*fpot(vmid))
        Z = Z + tri_area*mu
        Rcurrent[j,:] = mu*g
        Rrate = Rrate + np.sum(g**2)*mu*tri_area                     
    Rrate = Rrate/(Z*beta)
    Rcurrent = Rcurrent/(Z*beta) 
    # map reactive current on vertices
    Rcurrent_verts = np.zeros((Npts,2))
    tcount = np.zeros((Npts,1)) # the number of triangles adjacent to each vertex
    for j in range(Ntri):
        indt = np.array(tri[j,:])[:,None]    
        Rcurrent_verts[indt,:] = Rcurrent_verts[indt,:] + Rcurrent[j,:] # adds reactive current taken from center of each adjacent triangle to each vertice of the triangle
        tcount[indt] = tcount[indt] + 1   # updates the rows of tcount with indeces matching the entries in the indt array
    Rcurrent_verts = Rcurrent_verts/np.concatenate((tcount,tcount),axis = 1) # dividing to make it an average of neighboring triangles
    return Rcurrent_verts, Rrate

def trajectory_probability_density(pts,tri,fpot,beta,q):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)

    mu = np.zeros((Ntri,1))
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        vmid = np.reshape(np.sum(verts,axis=0)/3,(1,2))
        mu[j,:] = np.exp(-beta *fpot(vmid))

    # map reactive current on vertices
    mu_verts = np.zeros((Npts,1))
    tcount = np.zeros((Npts,1)) # the number of triangles adjacent to each vertex
    for j in range(Ntri):
        indt = np.array(tri[j,:])[:,None]    
        mu_verts[indt] = mu_verts[indt] + mu[j] # adds mu taken from center of each triangle to each vertice of the triangle
        tcount[indt] = tcount[indt] + 1   # adds a tri. to the count
    mu_verts = mu_verts/tcount # dividing to make it an average of neighboring triangles
    return mu_verts


# # parameters for the face potential
# xa=-3 
# ya=3
# xb=0 
# yb=4.5
# par = np.array([xa,ya,xb,yb]) # centers of sets A and B

# # problem setup: choose sets A and B and the outer boundary
# # set A is the circle with center at (xa,ya) and radius ra
# # set B is the circle with center at (xb,yb) and radius rb
# ra = 0.5 # radius of set A
# rb = 0.5 # radius of set B
# beta = 3 # beta = 1/(k_B T), T = temperature, k_B = Boltzmann's constant
# Vbdry = 12 # level set of the outer boundary {x : fpot(x) = Vbdry}

# # if generate_mesh = True, mesh is generated and saves as csv files
# # if generate_mesh = False, mesh is downloaded from those csv files
# generate_mesh = False

# # h0 is the desired scalind parameter for the mesh
# h0 = 0.1


# def face(xy,par):
#     xa=par[0]
#     ya=par[1]
#     xb=par[2] 
#     yb=par[3]
#     x = xy[:,0]
#     y = xy[:,1]
#     f=(1-x)**2+(y-0.25*x**2)**2+1
#     g1=1-np.exp(-0.125*((x-xa)**2+(y-ya)**2))
#     g2=1-np.exp(-0.25*(((x-xb)**2+(y-yb)**2)))
#     g3=1.2-np.exp(-2*((x+0)**2+(y-2)**2))
#     g4=1+np.exp(-2*(x+1.5)**2-(y-3.5)**2-(x+1)*(y-3.5))
#     v = f*g1*g2*g3*g4
#     return v


# def main():
    # pts = np.loadtxt('face_pts.csv', delimiter=',', dtype=float)
    # tri = np.loadtxt('face_tri.csv', delimiter=',', dtype=int)
    # def fpot(pts):
        # return face(pts,par)
    # find the mesh points lying on the Dirichlet boundary \partial A \cup \partial B
    # NAind,Aind = find_ABbdry_pts(pts,xa,ya,ra,h0) # find mesh points on \partial A
    # NBind,Bind = find_ABbdry_pts(pts,xb,yb,rb,h0) # find mesh points on \partial B

    # q = FEM_committor_solver(pts,tri,Aind,Bind,fpot,beta)

    # trajectory_probability_density(pts, tri, fpot, beta, q)

# if __name__ == "__main__":
    # main()