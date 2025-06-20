import os
import copy
import openmc
import numpy as np

def summarize_water_sph_statepoint(sp_path, label):
    sp = openmc.StatePoint(sp_path)
    transport_time = sp.runtime['transport']
    tally = sp.get_tally(name="flux tally")
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

def run_water_sph():

    #--------------------
    # model creation code
    #--------------------

    orig_dir = os.getcwd()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SCRIPT_DIR)
 

    water = openmc.Material(name='Water')
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cc', 1.0)
    water.add_s_alpha_beta('c_H_in_H2O')  

    materials = openmc.Materials([water])

    source_sphere = openmc.Sphere(r=0.5, name='Source sphere')
    water_sphere = openmc.Sphere(r=60.0, name='Water sphere')

    target_center = [62.0, 0.0, 0.0]
    target_sphere = openmc.Sphere(x0=target_center[0], y0=target_center[1], 
                                z0=target_center[2], r=0.5, name='Target sphere')

    boundary_min = -63.0
    boundary_max = 63.0
    x_min = openmc.XPlane(x0=boundary_min, boundary_type='vacuum')
    x_max = openmc.XPlane(x0=boundary_max, boundary_type='vacuum')
    y_min = openmc.YPlane(y0=boundary_min, boundary_type='vacuum')
    y_max = openmc.YPlane(y0=boundary_max, boundary_type='vacuum')
    z_min = openmc.ZPlane(z0=boundary_min, boundary_type='vacuum')
    z_max = openmc.ZPlane(z0=boundary_max, boundary_type='vacuum')

    source_cell = openmc.Cell(name='Source cell', fill=None)
    source_cell.region = -source_sphere

    water_cell = openmc.Cell(name='Water cell')
    water_cell.region = +source_sphere & -water_sphere
    water_cell.fill = water

    target_cell = openmc.Cell(name='Target cell', fill=None)
    target_cell.region = -target_sphere

    void_cell = openmc.Cell(name='Void region', fill=None)
    void_cell.region = (+water_sphere & +target_sphere &
                    +x_min & -x_max & +y_min & -y_max & +z_min & -z_max)

    geometry = openmc.Geometry([source_cell, water_cell, target_cell, void_cell])
    space = openmc.stats.Box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    energy = openmc.stats.Discrete([14.1e6], [1.0])
    source = openmc.IndependentSource(space=space, energy=energy, constraints={'domains':[source_cell]})

    settings = openmc.Settings()
    settings.source = source
    settings.batches = 30
    settings.particles = 1000000
    settings.run_mode = 'fixed source'

    target_filter = openmc.CellFilter(target_cell)
    tally = openmc.Tally(name="flux tally")
    tally.filters = [target_filter]
    tally.scores = ['flux']
    tallies = openmc.Tallies([tally])

    plot = openmc.Plot()
    plot.origin = [0, 0, 0]
    plot.width = [126, 126, 126]
    plot.pixels = [126, 126, 126]
    plot.type = 'voxel'
    
    plots = openmc.Plots([plot])
    model = openmc.Model()
    model.materials = materials
    model.geometry = geometry
    model.settings = settings
    model.tallies = tallies
    model.plots = plots
    
    #---------------------------
    # run_random_ray calculation 
    #--------------------------- 

    random_ray_model = copy.deepcopy(model)

    random_ray_model.convert_to_multigroup(
        method='material_wise', 
        correction='P0', 
        groups=openmc.mgxs.EnergyGroups(openmc.mgxs.GROUP_STRUCTURES['CASMO-2']), 
        nparticles=100000
    )

    random_ray_model.convert_to_random_ray()

    mesh = openmc.SphericalMesh(
        r_grid=np.linspace(0.0, 90.0, 90), 
        origin=[0.0, 0.0, 0.0]
    )

    random_ray_model.settings.random_ray['source_region_meshes'] = [(mesh, [geometry.root_universe])]
    random_ray_model.settings.random_ray['distance_inactive'] = 60.0
    random_ray_model.settings.random_ray['distance_active'] = 120.0
    random_ray_model.settings.random_ray['sample_method'] = 'prng'
    random_ray_model.settings.particles = 10000 #100000
    random_ray_model.settings.inactive = 50 #200
    random_ray_model.settings.batches = 100 #300

    wwg = openmc.WeightWindowGenerator(
        mesh, method='fw_cadis', max_realizations=random_ray_model.settings.batches)
    random_ray_model.settings.weight_window_generators = [wwg]

    random_ray_model.run(path='random_ray.xml')

    #-------------------
    # run_mc calculation
    #-------------------

    model.settings.weight_window_checkpoints = {'collision': True, 'surface': True}
    model.settings.survival_biasing = False
    wws = openmc.hdf5_to_wws('weight_windows.h5')
    model.settings.weight_windows = wws

    model.settings.particles = 20000
    model.settings.batches = 45
    
    model.settings.weight_windows_on = True
    statepoint_name = model.run(path='with_WW.xml')
    result_with_WW = summarize_water_sph_statepoint(statepoint_name, 'with_WW')

    model.settings.particles = 100000
    model.settings.batches = 60

    model.settings.weight_windows_on = False
    statepoint_name  = model.run(path='no_WW.xml')
    result_no_WW = summarize_water_sph_statepoint(statepoint_name, 'no_WW')

    os.chdir(orig_dir)

    return result_with_WW, result_no_WW

if __name__ == "__main__":
    run_water_sph()