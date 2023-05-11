import clump_generator
from time import perf_counter

t_ferelec = perf_counter()
clump_generator.Ferellec_McDowell(
    stl_dir="Rock.stl",
    rstep=0.001,
    rmin=0.01,
    dmin=0.01,
    pmax=1.0,
    output_dir='my_clump_fd',
    isShuffled=True)

print(f"Ferellec McDowell:\t{perf_counter() - t_ferelec}")


t_euclidean = perf_counter()
clump_generator.Euclidean_Distance_Transform(
    stl_dir="Rock.stl",
    N=24,
    rMin=0.0,
    div=102,
    overlap=0.6,
    output_dir='my_clump_edt')

print(f"Euclidean Distance Transform:\t{perf_counter() - t_euclidean}")


t_favier = perf_counter()
clump_generator.Favier(
    stl_dir="Rock.stl",
    chooseDistance="min",
    N=20,
    output_dir='my_clump_favier'
)
print(f"Favier:\t{perf_counter() - t_favier}")
