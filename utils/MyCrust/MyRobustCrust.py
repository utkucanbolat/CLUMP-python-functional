import numpy as np
import time
from scipy.spatial import Delaunay


def MyRobustCrust(p):
    # error check
    if len(p.shape) > 2 or p.shape[1] != 3:
        raise ValueError("Input 3D points must be stored in a Nx3 array")

    # Main

    # add points to the given ones, this is useful
    # to create outside tetrahedrons
    start = time.time()
    p, nshield = AddShield(p)
    print(f'Added Shield: {time.time() - start} s')

    start = time.time()
    tetr = Delaunay(p)
    tetr = tetr.simplices
    print(f'Delaunay Triangulation Time: {time.time() - start} s')

    # connectivity data
    # find triangles to tetrahedron and tetrahedron to triangles connectivity data
    start = time.time()
    t2tetr, tetr2t, t = Connectivity(tetr)
    print(f'Connectivity Time: {time.time() - start} s')

    start = time.time()
    cc, r = CC(p, tetr)  # Circumcenters of tetrahedrons
    print(f'Circumcenters Time: {time.time() - start} s')

    start = time.time()
    tbound, _ = Marking(p, tetr, tetr2t, t2tetr, cc, r, nshield)  # Flagging tetrahedrons as inside or outside
    print(f'Walking Time: {time.time() - start} s')

    # reconstruct raw surface
    t = t[tbound, :]
    del tetr, tetr2t, t2tetr

    # manifold extraction
    start = time.time()
    t, tnorm = ManifoldExtraction(t, p)
    print(f'Manifold extraction Time: {time.time() - start} s')

    return t, tnorm


def CC(p, tetr):
    ntetr = tetr.shape[0]
    cutsize = 25000
    i1 = 0
    i2 = cutsize
    r = np.zeros(ntetr)
    cc = np.zeros((ntetr, 3))

    if i2 > ntetr:
        i2 = ntetr

    while True:
        p1 = p[tetr[i1:i2, 0], :]
        p2 = p[tetr[i1:i2, 1], :]
        p3 = p[tetr[i1:i2, 2], :]
        p4 = p[tetr[i1:i2, 3], :]

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

        cc[i1:i2, 0] = detx / Det
        cc[i1:i2, 1] = dety / Det
        cc[i1:i2, 2] = detz / Det

        r[i1:i2] = np.sqrt(np.sum((p2 - cc[i1:i2, :]) ** 2, axis=1))

        if i2 == ntetr:
            break

        i1 += cutsize
        i2 += cutsize

        if i2 > ntetr:
            i2 = ntetr

    return cc, r


def Connectivity(tetr):
    # Gets connectivity relationships among tetrahedrons
    numt = tetr.shape[0]
    vect = np.arange(numt)
    t = np.vstack([tetr[:, [0, 1, 2]], tetr[:, [1, 2, 3]], tetr[:, [0, 2, 3]], tetr[:, [0, 1, 3]]])  # triangles not unique
    t, j = np.unique(np.sort(t, axis=1), axis=0, return_inverse=True)  # triangles
    t2tetr = np.column_stack([j[vect], j[vect + numt], j[vect + 2 * numt], j[vect + 3 * numt]])  # each tetrahedron has 4 triangles

    # triang-to-tetr connectivity
    nume = t.shape[0]
    tetr2t = np.zeros((nume, 2), dtype=np.int32)
    count = np.ones(nume, dtype=np.int8)

    for k in range(numt):
        for j in range(4):
            ce = t2tetr[k, j]
            tetr2t[ce, count[ce] - 1] = k  # Python is 0-indexed
            count[ce] += 1

    return t2tetr, tetr2t, t


def Marking(p, tetr, tetr2t, t2tetr, cc, r, nshield):
    # The more important routine to flag tetrahedroms as outside or inside

    # Constants for the algorithm
    TOLLDIFF = 0.01
    INITTOLL = 0.99
    MAXLEVEL = 10 / TOLLDIFF
    BRUTELEVEL = MAXLEVEL - 50

    # Preallocation
    NP = p.shape[0] - nshield
    numtetr = tetr.shape[0]
    nt = tetr2t.shape[0]
    onfront = np.zeros(nt, dtype=bool)

    # Flag as outside tetrahedrons with Shield points
    deleted = np.any(tetr > NP, axis=1)
    checked = deleted.copy()
    onfront[t2tetr[checked, :]] = True
    countchecked = np.sum(checked)

    # Tolerances to mark as in or out
    toll = np.zeros(nt) + INITTOLL
    level = 0

    # Intersection factor
    Ifact = IntersectionFactor(tetr2t, cc, r)

    # Now we scan all tetrahedrons.
    ids = np.arange(nt)
    queue = ids[onfront]
    nt = len(queue)

    while countchecked < numtetr and level < MAXLEVEL:
        level += 1

        for i in range(nt):
            temp_id = queue[i]
            tetr1, tetr2 = tetr2t[temp_id, :]

            if tetr2 == 0 or (checked[tetr1] and checked[tetr2]):
                onfront[temp_id] = False
                continue

            if Ifact[temp_id] >= toll[temp_id]:  # flag as equal
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
                onfront[temp_id] = False

            elif Ifact[temp_id] < -toll[temp_id]:  # flag as different
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
                onfront[temp_id] = False

            else:
                toll[temp_id] -= TOLLDIFF  # tolleraces were too high next time will be lower

        if level == BRUTELEVEL:  # brute continuation(this may appens when there are almost null volume tetrahedrons)
            print('Brute continuation necessary')
            onfront[t2tetr[~checked, :]] = True  # force onfront collocation

        # update the queue
        queue = ids[onfront]
        nt = len(queue)

    # Extract boundary triangles
    tbound = BoundTriangles(tetr2t, deleted)

    if level == MAXLEVEL:
        print(f'{level} th level was reached')

    return tbound, Ifact


def AddShield(p):
    # adds outside points to the given cloud forming outside tetrahedrons

    # find the bounding box
    maxx = np.max(p[:, 0])
    maxy = np.max(p[:, 1])
    maxz = np.max(p[:, 2])
    minx = np.min(p[:, 0])
    miny = np.min(p[:, 1])
    minz = np.min(p[:, 2])

    # give offset to the bounding box
    step = np.max(np.abs([maxx - minx, maxy - miny, maxz - minz]))

    maxx = maxx + step
    maxy = maxy + step
    maxz = maxz + step
    minx = minx - step
    miny = miny - step
    minz = minz - step

    N = 10  # number of points of the shield edge

    step = step / (N * N)  # decrease step, avoids not unique points

    nshield = N * N * 6

    # creating a grid lying on the bounding box
    vx = np.linspace(minx, maxx, N)
    vy = np.linspace(miny, maxy, N)
    vz = np.linspace(minz, maxz, N)

    # construct faces
    x, y = np.meshgrid(vx, vy)
    facez1 = np.column_stack((x.flatten(), y.flatten(), np.full((N * N,), maxz)))
    facez2 = np.column_stack((x.flatten(), y.flatten(), np.full((N * N,), minz)))

    x, y = np.meshgrid(vy, vz - step)
    facex1 = np.column_stack((np.full((N * N,), maxx), x.flatten(), y.flatten()))
    facex2 = np.column_stack((np.full((N * N,), minx), x.flatten(), y.flatten()))

    x, y = np.meshgrid(vx - step, vz)
    facey1 = np.column_stack((x.flatten(), np.full((N * N,), maxy), y.flatten()))
    facey2 = np.column_stack((x.flatten(), np.full((N * N,), miny), y.flatten()))

    # add points to the p array
    pnew = np.vstack((p, facex1, facex2, facey1, facey2, facez1, facez2))

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

    numt = t.shape[0]
    vect = np.arange(numt)  # Triangle indices
    # Edges - not unique
    e = np.vstack((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]))
    # Unique edges
    e, j = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True)
    te = np.column_stack((j[vect], j[vect + numt], j[vect + 2 * numt]))
    nume = e.shape[0]
    e2t = np.zeros((nume, 2), dtype=np.int32)

    ne = e.shape[0]
    # npoints = p.shape[0]

    count = np.zeros(ne, dtype=np.int32)  # Number of candidate triangles for each edge
    etmapc = np.zeros((ne, 4), dtype=np.int32)

    for i in range(numt):
        i1 = te[i, 0]
        i2 = te[i, 1]
        i3 = te[i, 2]

        etmapc[i1, count[i1] % 4] = i
        etmapc[i2, count[i2] % 4] = i
        etmapc[i3, count[i3] % 4] = i

        count[i1] += 1
        count[i2] += 1
        count[i3] += 1

    etmap = []
    for i in range(ne):
        etmap.append(etmapc[i, :count[i]].tolist())

    tkeep = np.zeros(numt, dtype=bool)  # No triangle selected initially

    # Start the front

    # Building the queue to store edges on the front that need to be studied
    efront = np.zeros(nume, dtype=np.int32)  # Estimated length of the queue

    # Initialize the front

    tnorm = Tnorm(p, t)  # Get triangle normals

    # Find the highest triangle
    t1 = np.argmax((p[t[:, 0], 2] + p[t[:, 1], 2] + p[t[:, 2], 2]) / 3)

    if tnorm[t1, 2] < 0:
        tnorm[t1, :] = -tnorm[t1, :]  # Points upwards

    tkeep[t1] = True  # First triangle selected
    efront[:3] = te[t1, :3]
    e2t[te[t1, :3], 0] = t1
    nf = 3  # Number of edges on the front

    while nf > 0:
        k = efront[nf - 1]  # ID of edge on the front

        if e2t[k, 1] > 0 or e2t[k, 0] < 1 or count[k] < 2:  # Edge is no longer on the front or has no candidate triangles
            nf -= 1
            continue  # Skip

        # Candidate triangles
        idtcandidate = etmap[k]

        t1 = e2t[k, 0]  # Triangle we come from

        # Get data structure
        #    p1
        #   / | \
        #  t1 p3  e1  p4 t2(idt)
        #   \ | /
        #    p2
        alphamin = np.inf  # Initialize
        ttemp = t[t1, :]
        etemp = e[k, :]
        p1 = etemp[0]
        p2 = etemp[1]
        p3 = ttemp[np.logical_and(ttemp != p1, ttemp != p2)]  # Third point ID

        for i in range(len(idtcandidate)):
            t2 = idtcandidate[i]
            if t2 == t1:
                continue

            ttemp = t[t2, :]
            p4 = ttemp[np.logical_and(ttemp != p1, ttemp != p2)]  # Third point ID

            # Calculate the angle between the triangles and take the minimum
            alpha, tnorm2 = TriAngle(p[p1, :], p[p2, :], p[p3, :], p[p4, :], tnorm[t1, :])

            if alpha < alphamin:
                alphamin = alpha
                idt = t2
                tnorm[t2, :] = tnorm2  # Restore orientation

        # Update front according to the idt triangle
        tkeep[idt] = True
        for j in range(3):
            ide = te[idt, j]

            if e2t[ide, 0] < 1:  # Is it the first triangle for the current edge?
                efront[nf] = ide
                nf += 1
                e2t[ide, 0] = idt
            else:  # No, it is the second one
                efront[nf] = ide
                nf += 1
                e2t[ide, 1] = idt

        nf -= 1  # To avoid advancing further in the queue and encountering zero

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
