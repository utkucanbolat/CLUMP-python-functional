import numpy as np
import time
import trimesh
from scipy.spatial import Delaunay
import pyvista as pv
import scipy.io
import warnings
import subprocess
import numpy.linalg as la


def MyRobustCrust(p):
    # error check
    if len(p.shape) > 2 or p.shape[1] != 3:
        raise ValueError("Input 3D points must be stored in a Nx3 array")

    # Turn p to numpy array. It is trimesh object. It should work without this but still...
    # And sort the array
    p = np.array(p)
    sorted_indices = np.argsort(p[:, 0])
    p = p[sorted_indices]

    # add points to the given ones, this is useful to create outside tetrahedrons
    start = time.time()
    p, nshield = AddShield(p)
    print(f'Added Shield: {time.time() - start} s')

    start = time.time()
    # https://stackoverflow.com/questions/36604172/difference-between-matlab-delaunayn-and-scipy-delaunay
    tetr = matlab_delaunayn(p)
    print(f'Delaunay Triangulation Time: {time.time() - start} s')

    # connectivity data
    # find triangles to tetrahedron and tetrahedron to triangles connectivity data
    start = time.time()

    """mat = scipy.io.loadmat('tetr.mat')
    tetr = np.array(mat['tetr'])"""

    t2tetr, tetr2t, t = Connectivity(tetr)  # Connectivity not working because of p. so delaunay needs to be corrected
    print(f'Connectivity Time: {time.time() - start} s')

    """mat = scipy.io.loadmat('p.mat')
    p = np.array(mat['p_circum'])"""

    start = time.time()
    cc, r = CC(p, tetr)  # Circumcenters of tetrahedrons
    print(f'Circumcenters Time: {time.time() - start} s')

    start = time.time()
    tbound, _ = Marking(p, tetr, tetr2t, t2tetr, cc, r, nshield)  # Flagging tetrahedrons as inside or outside
    print(f'Walking Time: {time.time() - start} s')

    # reconstructed raw surface
    t = t[tbound]
    # del tetr, tetr2t, t2tetr

    mat = scipy.io.loadmat('t.mat')
    t = np.array(mat['t'])

    t = t - 1  # INDEXING CONVENTION

    mat = scipy.io.loadmat('p.mat')
    p = np.array(mat['p'])

    start = time.time()
    t, tnorm = ManifoldExtraction(t, p)
    print(f'Manifold extraction Time: {time.time() - start} s')

    return t, tnorm


def CC(points, tetrahedra):
    # Initialize arrays for circumcenters and radii
    circumcenters = np.zeros((tetrahedra.shape[0], 3))
    radii = np.zeros(tetrahedra.shape[0])

    # Calculate circumcenter and radius for each tetrahedron
    for i, tetrahedron in enumerate(tetrahedra):
        # Extract the points of the tetrahedron
        p1, p2, p3, p4 = points[tetrahedron]

        # Compute vectors between points
        v21 = p1 - p2
        v31 = p3 - p1
        v41 = p4 - p1

        # Compute auxiliary determinants
        d1 = np.dot(v41, 0.5 * (p1 + p4))
        d2 = np.dot(v21, 0.5 * (p1 + p2))
        d3 = np.dot(v31, 0.5 * (p1 + p3))

        # Compute main determinant
        det23 = v21[1] * v31[2] - v21[2] * v31[1]
        det13 = v21[2] * v31[0] - v21[0] * v31[2]
        det12 = v21[0] * v31[1] - v21[1] * v31[0]

        Det = v41[0] * det23 + v41[1] * det13 + v41[2] * det12

        # Compute circumcenter coordinates
        circumcenter_x = (d1 * det23 + v41[1] * (d3 * v21[2] - d2 * v31[2]) + v41[2] * (d2 * v31[1] - d3 * v21[1])) / Det
        circumcenter_y = (v41[0] * (d3 * v21[2] - d2 * v31[2]) + d1 * det13 + v41[2] * (d2 * v31[0] - d3 * v21[0])) / Det
        circumcenter_z = (v41[0] * (d3 * v21[1] - d2 * v31[1]) + v41[1] * (d2 * v31[0] - d3 * v21[0]) + d1 * det12) / Det

        circumcenters[i] = [circumcenter_x, circumcenter_y, circumcenter_z]

        # Compute circumradius
        radii[i] = np.sqrt(np.sum((p2 - circumcenters[i]) ** 2))

    return circumcenters, radii


def Connectivity(tetr):
    # Gets connectivity relationships among tetrahedrons
    numt = tetr.shape[0]
    vect = np.arange(numt)
    t = np.vstack([tetr[:, [0, 1, 2]], tetr[:, [1, 2, 3]], tetr[:, [0, 2, 3]], tetr[:, [0, 1, 3]]])  # triangles not unique
    t, j = np.unique(np.sort(t, axis=1), return_inverse=True, axis=0)  # triangles
    t2tetr = np.vstack([j[vect], j[vect + numt], j[vect + 2 * numt], j[vect + 3 * numt]]).T  # each tetrahedron has 4 triangles

    # triang-to-tetr connectivity
    nume = t.shape[0]
    tetr2t = np.zeros((nume, 2), dtype=np.int32)
    count = np.ones(nume, dtype=np.int8)
    for k in range(numt):
        for j in range(4):
            ce = t2tetr[k, j]
            tetr2t[ce, count[ce] - 1] = k
            count[ce] += 1

    return t2tetr, tetr2t, t


def Marking(p, tetr, tetr2t, t2tetr, cc, r, nshield):
    # constants for the algorithm
    TOLLDIFF = .01
    INITTOLL = .99
    MAXLEVEL = 10 / TOLLDIFF
    BRUTELEVEL = MAXLEVEL - 50

    # preallocation
    np_ = p.shape[0] - nshield
    numtetr = tetr.shape[0]
    nt = tetr2t.shape[0]

    # First flag as outside tetrahedrons with Shield points
    deleted = np.any(tetr > np_, axis=1)
    checked = deleted.copy()
    onfront = np.zeros(nt, dtype=bool)

    for i in np.where(checked)[0]:
        onfront[t2tetr[i, :]] = True

    countchecked = np.sum(checked)

    # tolerances to mark as in or out
    toll = np.full(nt, INITTOLL)
    level = 0

    # intersection factor
    Ifact = IntersectionFactor(tetr2t, cc, r)

    ids = np.arange(nt)
    queue = ids[onfront]
    nt = len(queue)
    while countchecked < numtetr and level < MAXLEVEL:
        level += 1

        for i in range(nt):
            id_ = queue[i]
            tetr1, tetr2 = tetr2t[id_]

            if tetr2 == 0 or (checked[tetr1] and checked[tetr2]):
                onfront[id_] = False
                continue

            if Ifact[id_] >= toll[id_]:  # flag as equal
                if checked[tetr1]:
                    deleted[tetr2] = deleted[tetr1]
                    checked[tetr2] = True
                    countchecked += 1
                    onfront[t2tetr[tetr2]] = True
                else:
                    deleted[tetr1] = deleted[tetr2]
                    checked[tetr1] = True
                    countchecked += 1
                    onfront[t2tetr[tetr1]] = True
                onfront[id_] = False

            elif Ifact[id_] < -toll[id_]:  # flag as different
                if checked[tetr1]:
                    deleted[tetr2] = not deleted[tetr1]
                    checked[tetr2] = True
                    countchecked += 1
                    onfront[t2tetr[tetr2]] = True
                else:
                    deleted[tetr1] = not deleted[tetr2]
                    checked[tetr1] = True
                    countchecked += 1
                    onfront[t2tetr[tetr1]] = True
                onfront[id_] = False

            else:
                toll[id_] -= TOLLDIFF

        if level == BRUTELEVEL:
            print('Brute continuation necessary')
            onfront[np.any(t2tetr[~checked], axis=0)] = True

        queue = ids[onfront]
        nt = len(queue)

    # extract boundary triangles
    tbound = BoundTriangles(tetr2t, deleted)

    if level == MAXLEVEL:
        print(f'{level} th level was reached')
    else:
        print(f'{level} th level was reached')

    print(f'{countchecked / numtetr * 100:.4f} % of tetrahedrons were checked')

    return tbound, Ifact


def AddShield(p):
    # Find the bounding box
    maxx = np.max(p[:, 0])
    maxy = np.max(p[:, 1])
    maxz = np.max(p[:, 2])
    minx = np.min(p[:, 0])
    miny = np.min(p[:, 1])
    minz = np.min(p[:, 2])

    # Give offset to the bounding box
    step = np.max(np.abs([maxx - minx, maxy - miny, maxz - minz]))

    maxx = maxx + step
    maxy = maxy + step
    maxz = maxz + step
    minx = minx - step
    miny = miny - step
    minz = minz - step

    N = 10  # Number of points on the shield edge

    step = step / (N * N)  # Decrease step, avoids non-unique points

    nshield = N * N * 6

    # Creating a grid lying on the bounding box
    vx = np.linspace(minx, maxx, N)
    vy = np.linspace(miny, maxy, N)
    vz = np.linspace(minz, maxz, N)

    x, y = np.meshgrid(vx, vy)
    x = x.T.flatten()  # Transpose x and flatten
    y = y.T.flatten()  # Transpose y and flatten
    facez1 = np.column_stack([x, y, np.ones(N * N) * maxz])
    facez2 = np.column_stack([x, y, np.ones(N * N) * minz])

    x, y = np.meshgrid(vy, vz - step)
    x = x.T.flatten()  # Transpose x and flatten
    y = y.T.flatten()  # Transpose y and flatten
    facex1 = np.column_stack([np.ones(N * N) * maxx, x, y])
    facex2 = np.column_stack([np.ones(N * N) * minx, x, y])

    x, y = np.meshgrid(vx - step, vz)
    x = x.T.flatten()  # Transpose x and flatten
    y = y.T.flatten()  # Transpose y and flatten
    facey1 = np.column_stack([x, np.ones(N * N) * maxy, y])
    facey2 = np.column_stack([x, np.ones(N * N) * miny, y])

    # Add points to the p array
    pnew = np.concatenate((p, facex1, facex2, facey1, facey2, facez1, facez2), axis=0)

    return pnew, nshield


def BoundTriangles(tetr2t, deleted):
    # Extracts boundary triangles from a set tetr2t connectivity and form the
    # deleted vector which tells tetrahedrons that are marked as out

    nt = tetr2t.shape[0]  # Number of total triangles

    tbound = np.ones((nt, 2), dtype=bool)  # Initialize to keep shape in next operation

    ind = tetr2t > 0  # Avoid null index
    tbound[ind] = deleted[tetr2t[ind] - 1]  # Mark 1 for deleted 0 for kept tetrahedrons, -1 because Python uses 0-based indexing

    tbound = np.sum(tbound, axis=1) == 1  # Boundary triangles only have one tetrahedron

    return tbound


def IntersectionFactor(tetr2t, cc, r):
    nt = tetr2t.shape[0]
    Ifact = np.zeros((nt, 1))  # Intersection factor

    i = tetr2t[:, 1] > 0

    # Distance between circumcenters
    distcc = np.sum((cc[tetr2t[i, 0] - 1, :] - cc[tetr2t[i, 1] - 1, :]) ** 2, axis=1)

    # Intersection factor
    Ifact[i] = ((-distcc + r[tetr2t[i, 0] - 1].flatten() ** 2 + r[tetr2t[i, 1] - 1].flatten() ** 2) / (
            2 * r[tetr2t[i, 0] - 1].flatten() * r[tetr2t[i, 1] - 1].flatten()))[:, np.newaxis]

    return Ifact


def TriAngle(p1, p2, p3, p4, planenorm):
    v21 = p1 - p2
    v31 = p3 - p1
    tnorm1 = np.cross(v21, v31)
    tnorm1 /= la.norm(tnorm1)

    v41 = p4 - p1
    tnorm2 = np.cross(v21, v41)
    tnorm2 /= la.norm(tnorm2)
    alpha = np.dot(tnorm1, tnorm2)
    alpha = np.arccos(alpha)

    if np.dot(planenorm, p4 - p3) < 0:
        alpha = alpha + 2 * (np.pi - alpha)

    if np.dot(planenorm, tnorm1) > 0:
        tnorm2 = -tnorm2

    return alpha, tnorm2


def ManifoldExtraction(t, p):
    numt = t.shape[0]
    vect = np.arange(numt)
    e = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    e, j = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True)

    # Unique edges
    te = np.vstack([j[vect], j[vect + numt], j[vect + 2 * numt]]).T
    nume = e.shape[0]
    e2t = np.zeros((nume, 2), dtype=np.int32)
    ne = e.shape[0]
    np_ = p.shape[0]
    count = np.zeros(ne, dtype=np.int32)
    etmapc = np.zeros((ne, 4), dtype=np.int32)

    for i in range(numt):
        i1, i2, i3 = te[i, :]

        etmapc[i1, count[i1]] = i
        etmapc[i2, count[i2]] = i
        etmapc[i3, count[i3]] = i

        count[i1] += 1
        count[i2] += 1
        count[i3] += 1

    etmap = [etmapc[i, :count[i]] for i in range(ne)]

    tkeep = np.zeros(numt, dtype=bool)
    efront = np.zeros(nume, dtype=np.int32)

    tnorm = Tnorm(p, t)

    t1 = np.argmax((p[t[:, 0], 2] + p[t[:, 1], 2] + p[t[:, 2], 2]) / 3)

    if tnorm[t1, 2] < 0:
        tnorm[t1, :] *= -1  # SKIPPED. SOMEWHAT UNCESSARY

    tkeep[t1] = True
    efront[:3] = te[t1, :3]
    e2t[te[t1, :3], 0] = t1
    nf = 3

    co = 0
    # assuming TriAngle function has been defined properly somewhere
    while nf > 0:
        co += 1
        if co % 1000 == 0: print(co)

        if co == 10:
            break

        k = efront[nf]  # id edge on front

        if e2t[k, 1] > 0 or e2t[k, 0] < 1 or count[k] < 2:  # edge is no more on front or it has no candidates triangles
            nf -= 1
            continue  # skip

        # candidate triangles
        idtcandidate = etmap[k]

        t1 = e2t[k, 0]  # triangle we come from

        # get data structure
        alphamin = float('inf')  # initialize
        ttemp = t[t1, :]
        etemp = e[k, :]
        p1 = etemp[0]
        p2 = etemp[1]

        # p3 = ttemp[(ttemp != p1) & (ttemp != p2)][0]  # third point id
        p3 = ttemp[np.all([ttemp != p1, ttemp != p2], axis=0)][0]

        for i in range(len(idtcandidate)):
            t2 = idtcandidate[i]
            if t2 == t1:
                continue

            ttemp = t[t2, :]
            # p4 = ttemp[(ttemp != p1) & (ttemp != p2)][0]
            p4 = ttemp[np.all([ttemp != p1, ttemp != p2], axis=0)][0]  # third point id

            # calculate the angle between the triangles and take the minimum
            alpha, tnorm2 = TriAngle(p[p1, :], p[p2, :], p[p3, :], p[p4, :], tnorm[t1, :])

            if alpha < alphamin:
                alphamin = alpha
                idt = t2  # ??????
                tnorm[t2, :] = tnorm2  # restore orientation

        # update front according to idttriangle
        tkeep[idt] = True
        for j in range(3):
            ide = te[idt, j]

            if e2t[ide, 0] < 1:  # Is it the first triangle for the current edge?
                efront[nf] = ide
                nf += 1
                e2t[ide, 0] = idt
            else:  # no, it is the second one
                efront[nf] = ide
                nf += 1
                e2t[ide, 1] = idt

        nf -= 1  # to avoid running ahead in the queue and finding a zero

    t = t[tkeep, :]
    tnorm = tnorm[tkeep, :]

    return t, tnorm


def Tnorm(p, t):
    # Computes normalized normals of triangles

    v21 = p[t[:, 0]] - p[t[:, 1]]
    v31 = p[t[:, 2]] - p[t[:, 0]]

    tnorm1 = np.zeros(t.shape)

    tnorm1[:, 0] = v21[:, 1] * v31[:, 2] - v21[:, 2] * v31[:, 1]
    tnorm1[:, 1] = v21[:, 2] * v31[:, 0] - v21[:, 0] * v31[:, 2]
    tnorm1[:, 2] = v21[:, 0] * v31[:, 1] - v21[:, 1] * v31[:, 0]

    L = np.sqrt(np.sum(tnorm1 ** 2, axis=1))

    tnorm1 = tnorm1 / L[:, None]

    return tnorm1


def merge_duplicate_points(X):
    """Merge out points that have coincident location."""
    dupes_found = False
    num_init_points = X.shape[0]

    # Get unique rows and their indices
    _, idx_map = np.unique(X, axis=0, return_index=True)

    num_unique_points = len(idx_map)
    if num_init_points > num_unique_points:
        # Undo the sort to preserve the ordering of points
        idx_map.sort()
        X = X[idx_map]
        dupes_found = True

    return X, dupes_found, idx_map


def run_qhull(points):
    points_str = '\n'.join(' '.join(str(x) for x in point) for point in points)
    input_data = f"{len(points)}\n3\n{points_str}".encode()
    proc = subprocess.Popen(["./qhull", "d", "Qt", "Qbb", "Qc", "Fv"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    output, _ = proc.communicate(input_data)

    # convert output from bytes to string
    output_str = output.decode()
    # convert string to list of lists, and convert to integers
    output_list = [list(map(int, line.split()[1:])) for line in output_str.split('\n')[1:] if line]
    # convert list of lists to numpy array of integers
    output_array = np.array(output_list, dtype=int)

    return output_array


def matlab_delaunayn(x):
    if x is None:
        raise ValueError('Not Enough Inputs')

    n = x.shape[1]

    if n < 1:
        raise ValueError('X has Low Column Number')

    x, dupesfound, idxmap = merge_duplicate_points(x)

    if dupesfound:
        print('Warning: Duplicate Data Points')

    m, n = x.shape

    if m < n + 1:
        raise ValueError('Not Enough Points for Tessellation')

    if m == n + 1:
        t = np.arange(n + 1)

        # Enforce the orientation convention
        if n == 2 or n == 3:
            PredicateMat = np.ones((m, m))
            PredicateMat[:, :n] = x[t, :n]
            orient = np.linalg.det(PredicateMat)

            if n == 3:
                orient *= -1

            if orient < 0:
                t[1], t[2] = t[2], t[1]

        if dupesfound:
            t = idxmap[t]

        return t

    # t = Delaunay(x, qhull_options="Qt Qbb Qc")  # Scipy's qhull
    # t = t.simplices

    t = run_qhull(x)  # My qhull

    # Strip the zero volume simplices that may have been created by the presence of degeneracy
    mt, nt = t.shape
    v = np.ones(mt, dtype=bool)

    for i in range(mt):
        xa = x[t[i, :nt - 1]]
        xb = x[t[i, nt - 1]]

        val = np.abs(np.linalg.det(xa - xb))
        valtol = np.finfo(float).eps * np.max(np.abs(np.concatenate((xa.flatten(), xb.flatten()))))

        if val <= valtol:
            v[i] = False

    t = t[v]

    if dupesfound:
        t = idxmap[t]

    return t


if __name__ == "__main__":
    mat = scipy.io.loadmat('p.mat')
    p = np.array(mat['p'])

    faces, _ = MyRobustCrust(p)
