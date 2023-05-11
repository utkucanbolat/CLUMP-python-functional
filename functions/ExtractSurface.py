import subprocess
import numpy as np
from scipy.spatial import ConvexHull

"""
Tesselation of the surface of a clump into a surface mesh
2021 © V. Angelidakis, S. Nadimi, M. Otsubo, S. Utili.
"""


def sphereContact(sphere1, sphere2):
    """
    inContact: boolean: whether sphere1 and sphere2 intersect
    :param sphere1: [1 x 4] [x,y,z,r]:	test sphere 1
    :param sphere2: [1 x 4] [x,y,z,r]:	test sphere 2
    :return: boolean
    """
    d0 = np.linalg.norm(sphere2[0:3] - sphere1[0:3])  # Centroidal distance of the spheres
    if sphere1[3] + sphere2[3] > d0 > np.abs(sphere1[3] - sphere2[3]):
        return True
    else:
        return False


def makeSphere(X, Y, Z, radius, N):
    """
    Function to create a surface mesh of a sphere with radius r, centered at (x,y,z) with N vertices.
    :param X:
    :param Y:
    :param Z:
    :param radius:
    :param N:
    :return: vertices/faces
    """
    vertices = np.zeros((N, 3))
    inc = np.pi * (3 - np.sqrt(5))
    off = 2 / N
    for k in range(N):
        y = k * off - 1 + off / 2
        r = np.sqrt(1 - y ** 2)
        phi = k * inc
        vertices[k, 0:3] = np.array([np.cos(phi) * r * radius, y * radius, np.sin(phi) * r * radius])

    vertices += [X, Y, Z]
    faces = ConvexHull(vertices).simplices  # this works but does not give the exact results with the MATLAB implementation

    return vertices, faces


def spherePotential(point, sphere, allowZero):
    """
    :param point: [1 x 3] x,y,z:	test point
    :param sphere: [1 x 4] x,y,z,r	sphere of interest
    :param allowZero: boolean: whether to consider 0 values as contact, i.e. returning true
    :return: isInside: boolean: whether the test point is inside the sphere of interest
    """
    if allowZero:
        isInside = np.sqrt(((sphere[0] - point[0]) ** 2 + (sphere[1] - point[1]) ** 2 + (sphere[2] - point[2]) ** 2) / (sphere[3] ** 2)) - 1 <= 0
    else:
        isInside = np.sqrt(((sphere[0] - point[0]) ** 2 + (sphere[1] - point[1]) ** 2 + (sphere[2] - point[2]) ** 2) / (sphere[3] ** 2)) - 1 < 0

    return isInside


def ExtractSurface(clump, N_sphere, N_circle):
    """
    :param clump: either "clump" object or N x 4 matrix with columns of [x,y,z,r], where x,y,z the centroid of each sphere and r its radius
    :param N_sphere: Number of vertices on the surface of each member-sphere of the clump
    :param N_circle: Number of vertices on the circle defined as the intersection of two overlapping spheres
    :return: faces: faces of generated surface mesh
             vertices : vertices of generated surface mesh
    """

    # Utku - skipped the check format of input

    ################################################################################################
    #                                   Main Body of the Function                                  #
    ################################################################################################

    # the matlab function deal maps right hand side to left. it simply is:
    x, y, z, r = zip(*clump)
    spheresList = clump

    # Contact detection between all spheres (all possible combinations) - Record interactions
    interactions = []
    for i in range(spheresList.shape[0] - 1):
        for j in range(i + 1, spheresList.shape[0]):
            if i == j:
                continue
            inContact = sphereContact(spheresList[i, :], spheresList[j, :])
            if inContact:
                interactions.append([i + 1, j + 1])

    interactions = np.array(interactions) - 1  # -1 IS CRUCIAL BECAUSE PYTHON INDICES STARTS FROM 0!

    # Generate points for each sphere
    S_struct = []
    for i in range(spheresList.shape[0]):
        S_vertices, S_faces = makeSphere(x[i], y[i], z[i], r[i], N_sphere)
        S_dict = {"vertices": S_vertices, "faces": S_faces}
        S_struct.append(S_dict)

    # Calculate points on the intersection of each pair of interacting spheres
    for i in range(interactions.shape[0]):
        n = spheresList[interactions[i, 1], :3] - spheresList[interactions[i, 0], :3]  # (not normalised) normal vector of each interaction
        d = np.linalg.norm(n)  # centroidal distance between sphere1-sphere2 in each interaction
        n = n / np.linalg.norm(n)  # normalised normal vector of each interaction

        r1 = spheresList[interactions[i, 0], 3]  # radius of sphere1
        r2 = spheresList[interactions[i, 1], 3]  # radius of sphere2

        h = np.sqrt((2 * r1 * d) ** 2 - (r1 ** 2 + d ** 2 - r2 ** 2) ** 2) / (2 * d)  # Radius of intersection circle
        alph = np.arccos((r1 ** 2 + d ** 2 - r2 ** 2) / (2 * r1 * d))
        h1 = r1 * (1 - np.cos(alph))
        C = spheresList[interactions[i, 0], 0:3] + n * (r1 - h1)  # Contact point

        n3 = n
        n1 = np.array([n[2], 0, -n[0]])  # Vector perpendicular to n

        if np.linalg.norm(n1) == 0:
            n1 = np.array([n[1], 0, -n[0]])

        n1 = n1 / np.linalg.norm(n1)  # Normalise n1
        n2 = np.cross(n3, n1)

        # Generate points of intersection circle
        step = 2 * np.pi / N_circle
        a = np.arange(-np.pi, np.pi + step, step)
        px = C[0] + h * (n1[0] * np.cos(a) + n2[0] * np.sin(a))
        py = C[1] + h * (n1[1] * np.cos(a) + n2[1] * np.sin(a))
        pz = C[2] + h * (n1[2] * np.cos(a) + n2[2] * np.sin(a))

        circlevertices = np.transpose(np.array([px, py, pz]))

        S_struct[i]["circlevertices"] = circlevertices

        S_struct[interactions[i, 0]]['circlevertices'] = circlevertices
        S_struct[interactions[i, 0]]['vertices'] = np.vstack(
            (S_struct[interactions[i, 0]]['vertices'],
             S_struct[interactions[i, 0]]['circlevertices']))

    # Perform contact detection to detect and delete points of each sphere
    for i in range(interactions.shape[0]):
        # For interaction [sphere1, sphere2], check which vertices of sphere1 are inside sphere2
        for j in range(S_struct[interactions[i, 0]]['vertices'].shape[0] - 1, -1, -1):
            if spherePotential(S_struct[interactions[i, 0]]['vertices'][j, :], spheresList[interactions[i, 1], :], True):
                S_struct[interactions[i, 0]]['vertices'] = np.delete(S_struct[interactions[i, 0]]['vertices'], j, axis=0)

        # For interaction [sphere1, sphere2], check which vertices of sphere2 are inside sphere1
        for j in range(S_struct[interactions[i, 1]]['vertices'].shape[0] - 1, -1, -1):
            if spherePotential(S_struct[interactions[i, 1]]['vertices'][j, :], spheresList[interactions[i, 0], :], True):
                S_struct[interactions[i, 1]]['vertices'] = np.delete(S_struct[interactions[i, 1]]['vertices'], j, axis=0)

    # Collect vertices from all spheres in one variable
    vertices = np.empty((0, 3))
    for i in range(len(S_struct)):
        vertices = np.vstack((vertices, S_struct[i]['vertices']))
    vertices = np.unique(vertices, axis=0).real

    np.savetxt("temp.csv", vertices, delimiter=",")  # write the vertices for the binary to read

    # Generate mesh using the Crust algorithm (Amenta et al, 1999)
    # Since the Crust algorithm has not yet been implemented in Python when this code was being implemented, I used the binaries of the powercrust algorithm
    binary_path = "../utils/crust_algorithm"
    file_name = "temp.csv"
    subprocess.run(["sudo", "-S", binary_path + "/run_MyRobustCrust.sh", file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                   input='password\n')

    # read faces
    faces = np.genfromtxt(file_name, delimiter=',')

    return faces, vertices


N_sphere = 400
N_circle = 200
clump = np.array([[1, 0, 0, 1.1],
                  [2, 1, 0, 1.1],
                  [3, 0, 0, 1.2]])
faces, vertices = ExtractSurface(clump, N_sphere, N_circle)

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Define the vertices and faces of the 3D image


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=faces, color='grey', alpha=0.5)
# check the triangles array for out-of-bounds vertex indices
min_vertex_index = np.min(faces)
max_vertex_index = np.max(faces)
if min_vertex_index < 0 or max_vertex_index >= len(vertices):
    raise ValueError('triangles array contains out-of-bounds vertex indices')
plt.show()
