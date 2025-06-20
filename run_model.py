from labyrinth_example.labyrinth      import run_labyrinth
from ITER_cylinder.ITER_cyl           import run_ITER_cyl
from water_sphere.water_sph           import run_water_sph

problems = [
    # ("Labyrinth",     run_labyrinth),
    # ("ITER_Cylinder", run_ITER_cyl),
    ("water_sphere", run_water_sph),
]

print()

results = {name: fn() for name, fn in problems}
for name, (WW, noWW) in results.items():
    label = name.lower()
    print(f"-- {label} FOM --")
    for mode, res in (("with_WW", WW), ("no_WW", noWW)):
        print(f"{mode} -------------------------------")
        print(f"  Avg σ       : {res['avg_rel_sigma']:.3e}")
        print(f"  Max σ       : {res['max_rel_sigma']:.3e}")
        print(f"  Transport T.: {res['transport_time']:.3f} s")
        print(f"  1 / σ²T      : {res['figure_of_merit']:.3e}")
        print()