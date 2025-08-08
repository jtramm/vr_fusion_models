import os
import openmc
import numpy as np

def summarize_ITER_cyl_statepoint(sp_path):
    sp = openmc.StatePoint(sp_path)
    transport_time = sp.runtime['transport']
    tally = sp.get_tally(name="tally_cells_n")
    means = tally.get_values(value='mean').ravel()
    means_safe = np.where(means == 0, 1.0, means)
    sigmas = tally.get_values(value='std_dev').ravel()
    sigmas_safe = np.where(sigmas == 0, 1.0, sigmas)
    
    avg_rel_sigma = np.mean(sigmas_safe / means_safe)
    max_rel_sigma = np.max(sigmas_safe / means_safe)
    figure_of_merit = 1 / (avg_rel_sigma**2 * transport_time)

    results = {}
    results['transport_time'] = transport_time
    results['avg_rel_sigma'] = avg_rel_sigma
    results['max_rel_sigma'] = max_rel_sigma
    results['figure_of_merit'] = figure_of_merit

    return results

def run_ITER_cyl(
        random_ray_edges=[0, 6.25e-1, 2e7], 
        weight_window_edges=[0, 6.25e-1, 2e7], 
        mesh_cell_size_cm=8, 
        MGXS_correction=None, 
        volume_estimator='naive',
    ): 

    print("\n" + "="*40)
    message = ">>> SIMULATING ITER_cyl <<<"
    for code in [196, 208, 226, 46, 21, 93, 201]:
        print(f"\033[1;38;5;{code}m{message}\033[0m") 
    print("="*40 + "\n")
    print()

    orig_dir = os.getcwd()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SCRIPT_DIR)
    
    if os.path.exists("mgxs.h5"):
        os.remove("mgxs.h5")
    
    random_ray_groups = openmc.mgxs.EnergyGroups(list(random_ray_edges))

    model = openmc.Model.from_model_xml("monte_carlo_ITER_cyl.xml")
    model.convert_to_multigroup(
        method="stochastic_slab",
        nparticles=10000,
        groups=random_ray_groups,
        correction=MGXS_correction,
    )

    model.convert_to_random_ray()

    mesh = openmc.RegularMesh()
    bbox = model.geometry.bounding_box
    ll   = np.array(bbox.lower_left)
    ur   = np.array(bbox.upper_right)
    mesh.lower_left = ll
    mesh.upper_right = ur
    dims = np.ceil((ur - ll) / mesh_cell_size_cm).astype(int)
    mesh.dimension = tuple(dims)

    model.settings.random_ray["source_region_meshes"] = [(mesh, [model.geometry.root_universe])]
    model.settings.random_ray['volume_estimator'] = volume_estimator
    model.settings.random_ray["distance_inactive"] = 1500.0
    model.settings.random_ray["distance_active"] = 3000.0
    model.settings.particles = 20000
    model.settings.batches = 200
    model.settings.inactive = 100

    wwg = openmc.WeightWindowGenerator(
        mesh, method='fw_cadis', energy_bounds=list(weight_window_edges), max_realizations=model.settings.batches)
    model.settings.weight_window_generators = [wwg]

    model.run(path='random_ray.xml')

    model = openmc.Model.from_model_xml("monte_carlo_ITER_cyl.xml")
    model.settings.weight_window_checkpoints = {'collision': True, 'surface': True}
    model.settings.survival_biasing = False
    wws = openmc.hdf5_to_wws('weight_windows.h5')
    model.settings.weight_windows = wws

    for tally in model.tallies:
        for flt in tally.filters:
            if isinstance(flt, openmc.MeshFilter):
                flt.mesh.lower_left  = ll
                flt.mesh.upper_right = ur
                flt.mesh.dimension   = tuple(dims)

    model.settings.particles = 100000
    model.settings.batches = 35
    model.settings.weight_windows_on = True
    statepoint_name = model.run(path='mc.xml')
    results_with_WW = summarize_ITER_cyl_statepoint(statepoint_name)

    model.settings.particles = 100000
    model.settings.batches = 70
    model.settings.weight_windows_on = False
    statepoint_name = model.run(path='mc.xml')
    results_no_WW = summarize_ITER_cyl_statepoint(statepoint_name)

    os.chdir(orig_dir)

    return results_with_WW, results_no_WW

if __name__ == "__main__":
    run_ITER_cyl()