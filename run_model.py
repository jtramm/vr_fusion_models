import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "labyrinth_example"))
sys.path.insert(0, os.path.join(HERE, "ITER_cylinder"))

from labyrinth_example.labyrinth      import run_labyrinth
from ITER_cylinder.ITER_cyl           import run_ITER_cyl

def main():
    os.chdir(os.path.join(HERE, "labyrinth_example"))
    WW_results_of_labyrinth, no_WW_results_of_labyrinth = run_labyrinth()
    os.chdir(os.path.join(HERE, "ITER_cylinder"))
    WW_results_of_ITER_cyl, no_WW_results_of_ITER_cyl = run_ITER_cyl()

    print()
    print("-- labyrinth FOM --")
    print("with_WW -------------------------------")
    print(f"  Avg σ       : {WW_results_of_labyrinth['avg_sigma']:.3e}")
    print(f"  Max σ       : {WW_results_of_labyrinth['max_sigma']:.3e}")
    print(f"  Transport T.: {WW_results_of_labyrinth['transport_time']:.3f} s")
    print(f"  1 / σ²T      : {WW_results_of_labyrinth['figure_of_merit']:.3e}")
    print()
    print("no_WW -------------------------------")
    print(f"  Avg σ       : {no_WW_results_of_labyrinth['avg_sigma']:.3e}")
    print(f"  Max σ       : {no_WW_results_of_labyrinth['max_sigma']:.3e}")
    print(f"  Transport T.: {no_WW_results_of_labyrinth['transport_time']:.3f} s")
    print(f"  1 / σ²T      : {no_WW_results_of_labyrinth['figure_of_merit']:.3e}")

    print()
    print("-- ITER Cylinder FOM --")
    print("with_WW -------------------------------")
    print(f"  Avg σ       : {WW_results_of_ITER_cyl['avg_sigma']:.3e}")
    print(f"  Max σ       : {WW_results_of_ITER_cyl['max_sigma']:.3e}")
    print(f"  Transport T.: {WW_results_of_ITER_cyl['transport_time']:.3f} s")
    print(f"  1 / σ²T      : {WW_results_of_ITER_cyl['figure_of_merit']:.3e}")
    print()
    print("no_WW -------------------------------")
    print(f"  Avg σ       : {no_WW_results_of_ITER_cyl['avg_sigma']:.3e}")
    print(f"  Max σ       : {no_WW_results_of_ITER_cyl['max_sigma']:.3e}")
    print(f"  Transport T.: {no_WW_results_of_ITER_cyl['transport_time']:.3f} s")
    print(f"  1 / σ²T      : {no_WW_results_of_ITER_cyl['figure_of_merit']:.3e}")

if __name__ == "__main__":
    main()