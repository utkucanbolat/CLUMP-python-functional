import numpy as np
import time
import trimesh
from scipy.spatial import Delaunay
import pyvista as pv
import scipy.io
import warnings


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

    t2tetr, tetr2t, t = Connectivity(tetr)  # Connectivity not working because of p. so delaunay needs to be corrected
    print(f'Connectivity Time: {time.time() - start} s')

    start = time.time()
    cc, r = CC(p, tetr)  # Circumcenters of tetrahedrons
    print(f'Circumcenters Time: {time.time() - start} s')

    start = time.time()
    tbound, _ = Marking(p, tetr, tetr2t, t2tetr, cc, r, nshield)  # Flagging tetrahedrons as inside or outside
    print(f'Walking Time: {time.time() - start} s')

    # reconstructed raw surface
    t = t[tbound]
    # del tetr, tetr2t, t2tetr

    start = time.time()
    t, tnorm = ManifoldExtraction(t, p)
    print(f'Manifold extraction Time: {time.time() - start} s')

    return t, tnorm


def CC(p, tetr):
    tetr = tetr - 1
    ntetr = tetr.shape[0]
    r = np.zeros(ntetr)
    cc = np.zeros((ntetr, 3))

    p1 = p[tetr[:, 0], :]
    p2 = p[tetr[:, 1], :]
    p3 = p[tetr[:, 2], :]
    p4 = p[tetr[:, 3], :]

    v21 = p1 - p2
    v31 = p3 - p1
    v41 = p4 - p1

    d1 = np.sum(v41 * (p1 + p4) * .5, axis=1)
    d2 = np.sum(v21 * (p1 + p2) * .5, axis=1)
    d3 = np.sum(v31 * (p1 + p3) * .5, axis=1)

    det23 = v21[:, 1] * v31[:, 2] - v21[:, 2] * v31[:, 1]
    det13 = v21[:, 2] * v31[:, 0] - v21[:, 0] * v31[:, 2]
    det12 = v21[:, 0] * v31[:, 1] - v21[:, 1] * v31[:, 0]

    Det = v41[:, 0] * det23 + v41[:, 1] * det13 + v41[:, 2] * det12

    detx = d1 * det23 + v41[:, 1] * (-(d2 * v31[:, 2]) + v21[:, 2] * d3) + v41[:, 2] * ((d2 * v31[:, 1]) - v21[:, 1] * d3)
    dety = v41[:, 0] * ((d2 * v31[:, 2]) - v21[:, 2] * d3) + d1 * det13 + v41[:, 2] * ((d3 * v21[:, 0]) - v31[:, 0] * d2)
    detz = v41[:, 0] * ((v21[:, 1] * d3) - d2 * v31[:, 1]) + v41[:, 1] * (d2 * v31[:, 0] - v21[:, 0] * d3) + d1 * det12

    cc[:, 0] = detx / Det
    cc[:, 1] = dety / Det
    cc[:, 2] = detz / Det

    r[:] = np.sqrt(np.sum((p2 - cc[:, :]) ** 2, axis=1))

    return cc, r


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
    TOLLDIFF = 0.01
    INITTOLL = 0.99
    MAXLEVEL = 10 / TOLLDIFF
    BRUTELEVEL = MAXLEVEL - 50

    n_p = p.shape[0] - nshield
    numtetr = tetr.shape[0]
    nt = tetr2t.shape[0]

    deleted = np.any(tetr > n_p, axis=1)
    checked = np.copy(deleted)
    onfront = np.zeros(nt, dtype=bool)
    onfront[t2tetr[checked]] = True
    countchecked = np.sum(checked)

    toll = np.zeros(nt) + INITTOLL
    level = 0

    Ifact = IntersectionFactor(tetr2t, cc, r)

    ids = np.arange(nt)
    queue = ids[onfront]
    nt = len(queue)
    while countchecked < numtetr and level < MAXLEVEL:
        level += 1
        for i in range(nt):
            id = queue[i]
            tetr1 = tetr2t[id, 0]
            tetr2 = tetr2t[id, 1]
            if tetr2 == 0:
                onfront[id] = False
                continue
            elif checked[tetr1] and checked[tetr2]:
                onfront[id] = False
                continue
            if Ifact[id] >= toll[id]:
                if checked[tetr1]:
                    deleted[tetr2] = deleted[tetr1]
                    checked[tetr2] = True
                    countchecked += 1
                    onfront[t2tetr[tetr2, :]] = True
                else:
                    deleted[tetr1] = deleted[tetr2]
                    checked[tetr1] = True
                    countchecked += 1
                    onfront[t2tetr[tetr1, :]] = True
                onfront[id] = False
            elif Ifact[id] < -toll[id]:
                if checked[tetr1]:
                    deleted[tetr2] = not deleted[tetr1]
                    checked[tetr2] = True
                    countchecked += 1
                    onfront[t2tetr[tetr2, :]] = True
                else:
                    deleted[tetr1] = not deleted[tetr2]
                    checked[tetr1] = True
                    countchecked += 1
                    onfront[t2tetr[tetr1, :]] = True
                onfront[id] = False
            else:
                toll[id] -= TOLLDIFF
        if level == BRUTELEVEL:
            warnings.warn('Brute continuation necessary')
            onfront[t2tetr[~checked, :]] = True
        queue = ids[onfront]
        nt = len(queue)

    tbound = BoundTriangles(tetr2t, deleted) # BoundTriangles is not defined in the provided code
    if level == MAXLEVEL:
        warnings.warn(f'{level} th level was reached')

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


def ManifoldExtraction(t, p):
    # Given a set of triangles,
    # Builds a manifold surface with the ball pivoting method.

    # building the etmap
    numt = t.shape[0]
    vect = np.arange(numt)  # Triangle indices
    e = np.vstack((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]))  # Edges - not unique
    e = np.unique(np.sort(e, axis=1), axis=0)  # Unique edges

    te = np.vstack((e[vect], e[vect + numt], e[vect + 2 * numt]))
    nume = e.shape[0]
    e2t = np.zeros((nume, 2), dtype=np.int32)

    ne = e.shape[0]
    # np_ = p.shape[0]

    count = np.zeros(ne, dtype=np.int32)  # numero di triangoli candidati per edge
    etmapc = np.zeros((ne, 4), dtype=np.int32)

    for i in range(numt):
        i1 = te[i, 0]
        i2 = te[i, 1]
        i3 = te[i, 2]

        etmapc[i1, 1 + count[i1]] = i
        etmapc[i2, 1 + count[i2]] = i
        etmapc[i3, 1 + count[i3]] = i

        count[i1] += 1
        count[i2] += 1
        count[i3] += 1

    etmap = [etmapc[i, :count[i]].tolist() for i in range(ne)]

    tkeep = np.full((numt,), False)  # all'inizio nessun trinagolo selezionato

    # Start the front
    # building the queue to store edges on front that need to be studied
    efront = np.zeros(nume, dtype=np.int32)  # exstimate length of the queue

    # Intilize the front
    tnorm = Tnorm(p, t)  # get triangles normals

    # find the highest triangle
    t1 = np.argmax((p[t[:, 0], 2] + p[t[:, 1], 2] + p[t[:, 2], 2]) / 3)

    if tnorm[t1, 2] < 0:
        tnorm[t1, :] = -tnorm[t1, :]  # punta verso l'alto

    tkeep[t1] = True  # primo triangolo selezionato
    efront[:3] = te[t1, :3]
    e2t[te[t1, :3], 0] = t1
    nf = 3  # efront iterato

    while nf > 0:
        k = efront[nf - 1]  # id edge on front

        if e2t[k, 1] > 0 or e2t[k, 0] < 1 or count[k] < 2:  # edge is no more on front or it has no candidates triangles
            nf -= 1
            continue  # skip

        # candidate triangles
        idtcandidate = etmap[k]

        t1 = e2t[k, 0]  # triangle we come from

        # get data structure
        # p1
        # / | \
        # t1 p3 e1 p4 t2(idt)
        # \ | /
        # p2
        alphamin = float('inf')  # inizilizza
        ttemp = t[t1, :]
        etemp = e[k, :]
        p1 = etemp[0]
        p2 = etemp[1]
        p3 = ttemp[(ttemp != p1) & (ttemp != p2)][0]  # terzo id punto

        for i in idtcandidate:
            t2 = i
            if t2 == t1:
                continue

            ttemp = t[t2, :]
            p4 = ttemp[(ttemp != p1) & (ttemp != p2)][0]  # terzo id punto

            # calcola l'angolo fra i triangoli e prendi il minimo
            alpha, tnorm2 = TriAngle(p[p1, :], p[p2, :], p[p3, :], p[p4, :], tnorm[t1, :])

            if alpha < alphamin:
                alphamin = alpha
                idt = t2
                tnorm[t2, :] = tnorm2  # ripristina orientazione

        # update front according to idttriangle
        tkeep[idt] = True
        for j in range(3):
            ide = te[idt, j]

            if e2t[ide, 0] < 1:  # Is it the first triangle for the current edge?
                efront[nf - 1] = ide
                nf += 1
                e2t[ide, 0] = idt
            else:  # no, it is the second one
                efront[nf - 1] = ide
                nf += 1
                e2t[ide, 1] = idt

        nf -= 1  # per evitare di scappare avanti nella coda e trovare uno zero

    t = t[tkeep, :]
    tnorm = tnorm[tkeep, :]
    return t, tnorm


def TriAngle(p1, p2, p3, p4, planenorm):
    # First, we see if p4 is above or below the plane identified
    # by the normal planenorm and the point p3

    test = np.sum(planenorm * p4 - planenorm * p3)

    # Computes angle between two triangles
    v21 = p1 - p2
    v31 = p3 - p1

    # normals to triangles
    tnorm1 = np.cross(v21, v31)
    tnorm1 /= np.linalg.norm(tnorm1)

    v41 = p4 - p1

    # normals to triangles
    tnorm2 = np.cross(v21, v41)
    tnorm2 /= np.linalg.norm(tnorm2)

    alpha = np.dot(tnorm1, tnorm2)  # cosine of the angle

    # The cosine considers the angle between the subplanes and not the triangles, it tells us
    # that the planes are at 180 if alpha = -1 they are in agreement if alpha = 1, at 90 °

    alpha = np.arccos(alpha)  # find the angle

    # If p4 is above the plane the angle is the right one otherwise it must be increased
    # by 2 * (180-alpha);
    if test < 0:  # p4 is under, we increase
        alpha += 2 * (np.pi - alpha)

    # We see if we need to change the orientation of the second triangle
    # as we have calculated them now tnorm1 t tnorm2 do not respect
    # orientation
    testor = np.sum(planenorm * tnorm1)
    if testor > 0:
        tnorm2 = -tnorm2

    return alpha, tnorm2


def Tnorm(p, t):
    # Computes normalized normals of triangles

    v21 = p[t[:, 0], :] - p[t[:, 1], :]
    v31 = p[t[:, 2], :] - p[t[:, 0], :]

    tnorm1 = np.empty_like(v21)
    tnorm1[:, 0] = v21[:, 1] * v31[:, 2] - v21[:, 2] * v31[:, 1]  # normals to triangles
    tnorm1[:, 1] = v21[:, 2] * v31[:, 0] - v21[:, 0] * v31[:, 2]
    tnorm1[:, 2] = v21[:, 0] * v31[:, 1] - v21[:, 1] * v31[:, 0]

    L = np.sqrt(np.sum(tnorm1 ** 2, axis=1))

    tnorm1[:, 0] /= L
    tnorm1[:, 1] /= L
    tnorm1[:, 2] /= L

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

    t = Delaunay(x, qhull_options="Qt Qbb Qc")

    # Strip the zero volume simplices that may have been created by the presence of degeneracy
    mt, nt = t.simplices.shape
    v = np.ones(mt, dtype=bool)

    for i in range(mt):
        xa = x[t.simplices[i, :nt - 1]]
        xb = x[t.simplices[i, nt - 1]]

        val = abs(np.linalg.det(xa - xb))
        valtol = np.finfo(float).eps * max(abs(np.concatenate((xa.flatten(), xb.flatten()))))

        if val <= valtol:
            v[i] = False

    t = t.simplices[v]

    if dupesfound:
        t = idxmap[t]

    return t


if __name__ == "__main__":
    mat = scipy.io.loadmat('p.mat')
    p = np.array(mat['p'])

    faces, _ = MyRobustCrust(p)
