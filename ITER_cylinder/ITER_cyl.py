import os
import openmc
import numpy as np

def summarize_ITER_cyl_statepoint(sp_path, label):
    sp = openmc.StatePoint(sp_path)
    transport_time = sp.runtime['transport']
    tally = sp.get_tally(id=1)
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

#---------------------------
# run_random_ray calculation 
#---------------------------

def run_ITER_cyl():

    orig_dir = os.getcwd()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SCRIPT_DIR)
    try:
        
        model = openmc.Model.from_model_xml("monte_carlo_ITER_cyl.xml")

        model.convert_to_multigroup(
            method="stochastic_slab",
            nparticles=10000
        )

        model.convert_to_random_ray()

        mesh = openmc.RegularMesh()
        bbox = model.geometry.bounding_box
        mesh.lower_left = bbox.lower_left
        mesh.upper_right = bbox.upper_right
        mesh.dimension = (120, 120, 196)


        model.settings.random_ray["source_region_meshes"] = [
            (mesh, [model.geometry.root_universe])
        ]
        model.settings.random_ray["distance_inactive"] = 1500.0
        model.settings.random_ray["distance_active"] = 3000.0
        model.settings.particles = 10000

        model.settings.batches = 100
        model.settings.inactive = 50

        wwg = openmc.WeightWindowGenerator(
            mesh, method='fw_cadis', max_realizations=model.settings.batches)
        model.settings.weight_window_generators = [wwg]

        plot = openmc.Plot()
        plot.origin = bbox.center
        plot.width = bbox.width
        plot.pixels = (200, 200, 160)
        plot.type = 'voxel'
        model.plots = [plot]

        model.run(path='random_ray.xml')

        #-------------------
        # run_mc calculation
        #-------------------

        model = openmc.Model.from_model_xml("monte_carlo_ITER_cyl.xml")
        model.settings.weight_window_checkpoints = {'collision': True, 'surface': True}
        model.settings.survival_biasing = False
        wws = openmc.hdf5_to_wws('weight_windows.h5')
        #wws[0].max_split = 10
        model.settings.weight_windows = wws

        model.settings.particles = 10
        model.settings.batches = 6

        model.settings.weight_windows_on = True
        statepoint_name = model.run(path='mc.xml')
        results_with_WW = summarize_ITER_cyl_statepoint(statepoint_name, "with_WW")

        model.settings.particles = 100000 
        model.settings.batches = 70

        model.settings.weight_windows_on = False
        statepoint_name = model.run(path='mc.xml')
        results_no_WW = summarize_ITER_cyl_statepoint(statepoint_name, "no_WW")

        return results_with_WW, results_no_WW
    
    finally:
        os.chdir(orig_dir)

if __name__ == "__main__":
    run_ITER_cyl()