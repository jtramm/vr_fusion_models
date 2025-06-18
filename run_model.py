from labyrinth_example.labyrinth      import run_labyrinth
from ITER_cylinder.ITER_cyl           import run_ITER_cyl


problems = [
    ("Labyrinth",     run_labyrinth),
    ("ITER Cylinder", run_ITER_cyl),
]

results = {name: fn() for name, fn in problems}

for name, (WW, noWW) in results.items():
    for mode, res in (("with_WW", WW), ("no_WW", noWW)):
        print(f"{name} {mode:7} → "
              f"Avg σ={res['avg_rel_sigma']:.3e}, "
              f"Max σ={res['max_rel_sigma']:.3e}, "
              f"Transport T.={res['transport_time']:.3f}s, "
              f"1 / σ²T={res['figure_of_merit']:.3e}")