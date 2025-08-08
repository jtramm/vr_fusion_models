import os
import copy
import numpy as np
import openmc

def summarize_proxima_fusion_statepoint(sp_path):
    sp = openmc.StatePoint(sp_path)
    transport_time = sp.runtime['transport']
    tally = sp.get_tally(name="flux_mesh_tally")
    means = tally.get_values(value='mean').ravel()
    sigmas = tally.get_values(value='std_dev').ravel()
    means_safe = np.where(means == 0.0, 1.0, means)
    sigmas_safe = np.where(sigmas == 0.0, 1.0, sigmas)
    rel = sigmas_safe / means_safe

    avg_rel_sigma = np.mean(rel)
    max_rel_sigma = np.max(rel)
    figure_of_merit = 1 / (avg_rel_sigma**2 * transport_time)

    results = {}
    results['transport_time'] = transport_time
    results['avg_rel_sigma'] = avg_rel_sigma
    results['max_rel_sigma'] = max_rel_sigma
    results['figure_of_merit'] = figure_of_merit

    return results

def run_proxima_fusion(
        random_ray_edges=[0, 6.25e-1, 2e7], 
        weight_window_edges=[0, 6.25e-1, 2e7], 
        mesh_cell_size_cm=20, 
        MGXS_correction=None,
        volume_estimator='naive',
    ):

    print("\n" + "="*40)
    message = ">>> SIMULATING proxima_fusion <<<"
    for code in [196, 208, 226, 46, 21, 93, 201]:
        print(f"\033[1;38;5;{code}m{message}\033[0m") 
    print("="*40 + "\n")
    print()

    mesh_file = 'dagmc_surface_mesh.h5m'

    orig_dir = os.getcwd()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SCRIPT_DIR)
    
    if os.path.exists("mgxs.h5"):
        os.remove("mgxs.h5")
        
    random_ray_groups = openmc.mgxs.EnergyGroups(list(random_ray_edges))

    layer_1 = openmc.Material(name='layer_1')
    layer_1.add_nuclide('Fe56', 1.0, 'ao'); layer_1.set_density('g/cm3', 7.0)
    layer_2 = openmc.Material(name='layer_2')
    layer_2.add_nuclide('Li6', 0.9, 'ao'); layer_2.add_nuclide('Li7', 0.1, 'ao'); layer_2.set_density('g/cm3', 2.0)
    layer_3 = openmc.Material(name='layer_3')
    layer_3.add_nuclide('Fe56', 1.0, 'ao'); layer_3.set_density('g/cm3', 7.0)
    magnet  = openmc.Material(name='magnet')
    magnet.add_nuclide('Fe56', 1.0, 'ao'); magnet.set_density('g/cm3', 7.0)
    materials = openmc.Materials([layer_1, layer_2, layer_3, magnet])
    materials.export_to_xml()

    dag_universe  = openmc.DAGMCUniverse(mesh_file)
    root_universe = dag_universe.bounded_universe()
    geometry      = openmc.Geometry(root_universe)
    geometry.export_to_xml()

    mesh = openmc.RegularMesh()
    bbox = geometry.bounding_box
    ll   = np.array(bbox.lower_left)
    ur   = np.array(bbox.upper_right)
    mesh.lower_left  = ll
    mesh.upper_right = ur
    dims = np.ceil((ur - ll) / mesh_cell_size_cm).astype(int)
    mesh.dimension   = tuple(dims)

    tally_flux = openmc.Tally(name='flux_mesh_tally')
    tally_flux.filters = [openmc.MeshFilter(mesh)]
    tally_flux.scores  = ['flux']

    tally_mat  = openmc.Tally(name='H3_material_tally')
    tally_mat.filters = [openmc.MaterialFilter(layer_2)]; tally_mat.scores = ['H3-production']
    tally_heat = openmc.Tally(name='magnet_heating_material_tally')
    tally_heat.filters = [openmc.MaterialFilter(magnet)]; tally_heat.scores = ['heating']
    tallies = openmc.Tallies([tally_flux, tally_mat, tally_heat])

    src = openmc.IndependentSource()
    src.space  = openmc.stats.CylindricalIndependent(
        r   = openmc.stats.Discrete([1397.626385546529], [1]),
        phi = openmc.stats.Uniform(0.0, 2.0 * np.pi),
        z   = openmc.stats.Discrete([0.0], [1])
    )
    src.angle  = openmc.stats.Isotropic()
    src.energy = openmc.stats.Discrete([14.06e6], [1])

    settings = openmc.Settings()
    settings.run_mode         = 'fixed source'
    settings.source           = src
    settings.photon_transport = False
    settings.cross_sections   = os.environ['OPENMC_CROSS_SECTIONS']

    model = openmc.Model(geometry=geometry, materials=materials, settings=settings, tallies=tallies)

    for tally in model.tallies:
        for flt in tally.filters:
            if isinstance(flt, openmc.MeshFilter):
                flt.mesh.lower_left  = ll
                flt.mesh.upper_right = ur
                flt.mesh.dimension   = tuple(dims)

    random_ray_model = copy.deepcopy(model)
    random_ray_model.tallies = openmc.Tallies()
    random_ray_model.settings.particles = 20000
    random_ray_model.settings.batches   = 200
    random_ray_model.settings.inactive  = 100
    random_ray_model.convert_to_multigroup(
        method="stochastic_slab",
        nparticles=10000,
        groups=random_ray_groups,
        correction=MGXS_correction,
    )

    random_ray_model.convert_to_random_ray()

    bbox      = geometry.bounding_box
    box_src = openmc.stats.Box(bbox.lower_left, bbox.upper_right)
    rr_src  = openmc.IndependentSource(
        space       = box_src,
        angle       = openmc.stats.Isotropic(),
        energy      = src.energy,
        constraints = {'domains': [root_universe]}
    )

    random_ray_model.settings.source                   = rr_src
    random_ray_model.settings.random_ray['ray_source'] = rr_src
    random_ray_model.settings.random_ray['distance_inactive'] = 1500.0
    random_ray_model.settings.random_ray['distance_active']   = 3000.0
    random_ray_model.settings.random_ray['sample_method']     = 'prng'
    random_ray_model.settings.random_ray['volume_estimator'] = volume_estimator

    wwg = openmc.WeightWindowGenerator(
        mesh, method='fw_cadis', energy_bounds=list(weight_window_edges), max_realizations=random_ray_model.settings.batches)
    random_ray_model.settings.weight_window_generators = [wwg]
    
    random_ray_model.run(path='random_ray.xml')

    model.settings.weight_window_checkpoints = {'collision': True, 'surface': True}
    model.settings.survival_biasing          = False
    model.settings.weight_windows            = openmc.hdf5_to_wws('weight_windows.h5')

    model.settings.particles = 20000
    model.settings.batches   = 20
    model.settings.weight_windows_on = True
    sp_with = model.run(path='mc_with_ww.xml')
    results_with_WW = summarize_proxima_fusion_statepoint(sp_with)

    model.settings.particles = 20000
    model.settings.batches   = 20
    model.settings.weight_windows_on = False
    sp_no = model.run(path='mc_no_ww.xml')
    results_no_WW = summarize_proxima_fusion_statepoint(sp_no)

    os.chdir(orig_dir)
    return results_with_WW, results_no_WW

if __name__ == '__main__':
    run_proxima_fusion()