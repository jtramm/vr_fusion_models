import os
import copy
import openmc
import numpy as np

def summarize_JETSON_2D_statepoint(sp_path):
    sp = openmc.StatePoint(sp_path)
    transport_time = sp.runtime['transport']
    tally = sp.get_tally(name="mesh_flux")
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

def run_JETSON_2D(
        random_ray_edges=[0, 6.25e-1, 2e7], 
        weight_window_edges=[0, 6.25e-1, 2e7], 
        mesh_cell_size_cm=10, 
        MGXS_correction=None, 
        volume_estimator='naive'
    ):

    print("\n" + "="*40)
    message = ">>> SIMULATING JETSON_2D <<<"
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

    mat_inconel = openmc.Material(name='Inconel600')
    mat_inconel.add_element('Ni', 0.75)
    mat_inconel.add_element('Fe', 0.10)
    mat_inconel.add_element('Cr', 0.15)
    mat_inconel.set_density('g/cm3', 8.0)

    mat_steel = openmc.Material(name='SS304B7')
    mat_steel.add_element('Fe', 0.67)
    mat_steel.add_element('Cr', 0.20)
    mat_steel.add_element('Ni', 0.12)
    mat_steel.add_nuclide('B10', 0.01)
    mat_steel.set_density('g/cm3', 7.8)

    mat_water = openmc.Material(name='H2O')
    mat_water.add_nuclide('H1', 2)
    mat_water.add_nuclide('O16', 1)
    mat_water.set_density('g/cm3', 1.0)

    mat_concrete = openmc.Material(name='Concrete')
    mat_concrete.set_density('g/cm3', 2.3)
    mat_concrete.add_nuclide('H1', 0.20)
    mat_concrete.add_nuclide('O16', 0.10)
    mat_concrete.add_element('Si', 0.24)
    mat_concrete.add_element('Ca', 0.18)
    mat_concrete.add_element('Al', 0.02)
    mat_concrete.add_element('C', 0.01)

    borated_concrete = openmc.Material(name='BoroConcrete')

    borated_concrete.add_nuclide('Si28', 0.185, 'wo')  
    borated_concrete.add_nuclide('Si29', 0.0097, 'wo') 
    borated_concrete.add_nuclide('Si30', 0.0063, 'wo') 

    borated_concrete.add_nuclide('Ca40', 0.158, 'wo') 
    borated_concrete.add_nuclide('Ca42', 0.00105, 'wo')
    borated_concrete.add_nuclide('Ca43', 0.00022, 'wo') 
    borated_concrete.add_nuclide('Ca44', 0.00340, 'wo') 
    borated_concrete.add_nuclide('Ca46', 0.0000065, 'wo') 
    borated_concrete.add_nuclide('Ca48', 0.00030, 'wo') 

    borated_concrete.add_nuclide('Al27', 0.045, 'wo')  

    borated_concrete.add_nuclide('Fe54', 0.0032, 'wo') 
    borated_concrete.add_nuclide('Fe56', 0.0502, 'wo') 
    borated_concrete.add_nuclide('Fe57', 0.0012, 'wo') 
    borated_concrete.add_nuclide('Fe58', 0.00015, 'wo') 

    borated_concrete.add_nuclide('Mg24', 0.0079, 'wo') 
    borated_concrete.add_nuclide('Mg25', 0.0010, 'wo') 
    borated_concrete.add_nuclide('Mg26', 0.0011, 'wo') 

    borated_concrete.add_nuclide('K39', 0.0185, 'wo')   
    borated_concrete.add_nuclide('K40', 0.0000023, 'wo')
    borated_concrete.add_nuclide('K41', 0.00134, 'wo')

    borated_concrete.add_nuclide('Na23', 0.015, 'wo') 

    borated_concrete.add_element('O', 0.455, 'wo')

    borated_concrete.add_nuclide('H1', 0.008, 'wo')   
    borated_concrete.add_nuclide('H2', 0.0000012, 'wo')

    borated_concrete.add_nuclide('C12', 0.0099, 'wo')  
    borated_concrete.add_nuclide('C13', 0.00011, 'wo')

    borated_concrete.add_nuclide('B10', 0.00199, 'wo') 
    borated_concrete.add_nuclide('B11', 0.00801, 'wo')

    borated_concrete.add_nuclide('S32', 0.00095, 'wo') 
    borated_concrete.add_nuclide('S33', 0.0000076, 'wo')
    borated_concrete.add_nuclide('S34', 0.000043, 'wo')
    borated_concrete.add_nuclide('S36', 0.0000000002, 'wo') 

    borated_concrete.add_nuclide('Ti46', 0.000082, 'wo') 
    borated_concrete.add_nuclide('Ti47', 0.000074, 'wo') 
    borated_concrete.add_nuclide('Ti48', 0.00073, 'wo')  
    borated_concrete.add_nuclide('Ti49', 0.000054, 'wo') 
    borated_concrete.add_nuclide('Ti50', 0.000052, 'wo') 

    borated_concrete.add_nuclide('Mn55', 0.0005, 'wo')

    mat_air = openmc.Material(name="Air")
    mat_air.set_density('g/cm3', 0.001225)  
    mat_air.add_element('N', 0.78084, 'ao')  
    mat_air.add_element('O', 0.20946, 'ao')  
    mat_air.add_element('Ar', 0.00934, 'ao')
    mat_air.add_element('C', 0.00036, 'ao') 

    mats = openmc.Materials([mat_inconel, mat_steel, mat_water,
                            mat_concrete, borated_concrete, mat_air])

    R0_plasma = 296.0  
    a_plasma = 210.0   
    cwt = a_plasma + 1.5  
    st = cwt + 22        
    ct = st + 18       
    owti = ct + 1.5    

    r_inner_air_outer = R0_plasma - owti    
    r_inner_wrapper_outer = R0_plasma - ct  
    r_inner_coolant_outer = R0_plasma - st  
    r_inner_shield_outer = R0_plasma - cwt 
    r_inner_chamber_outer = R0_plasma - a_plasma 

    r_outer_chamber_inner = R0_plasma + a_plasma  
    r_outer_shield_inner = R0_plasma + cwt      
    r_outer_coolant_inner = R0_plasma + st     
    r_outer_wrapper_inner = R0_plasma + ct   
    r_outer_air_inner = R0_plasma + owti       

    inner_air_outer = openmc.ZCylinder(r=r_inner_air_outer)
    inner_wrapper_outer = openmc.ZCylinder(r=r_inner_wrapper_outer)
    inner_coolant_outer = openmc.ZCylinder(r=r_inner_coolant_outer)
    inner_shield_outer = openmc.ZCylinder(r=r_inner_shield_outer)
    inner_chamber_outer = openmc.ZCylinder(r=r_inner_chamber_outer)

    outer_chamber_inner = openmc.ZCylinder(r=r_outer_chamber_inner)
    outer_shield_inner = openmc.ZCylinder(r=r_outer_shield_inner)
    outer_coolant_inner = openmc.ZCylinder(r=r_outer_coolant_inner)
    outer_wrapper_inner = openmc.ZCylinder(r=r_outer_wrapper_inner)
    outer_air_inner = openmc.ZCylinder(r=r_outer_air_inner)

    li = 1720  
    lo = 1750 
    ro = 2000 

    room_liner_inner = openmc.model.RectangularParallelepiped(
        xmin=-li, xmax=li,
        ymin=-li, ymax=li,
        zmin=-li, zmax=li
    )

    room_liner_outer = openmc.model.RectangularParallelepiped(
        xmin=-lo, xmax=lo,
        ymin=-lo, ymax=lo,
        zmin=-lo, zmax=lo
    )

    x_min = openmc.XPlane(-ro, boundary_type='vacuum')
    x_max = openmc.XPlane(ro, boundary_type='vacuum')
    y_min = openmc.YPlane(-ro, boundary_type='vacuum')
    y_max = openmc.YPlane(ro, boundary_type='vacuum')

    central_air_cell = openmc.Cell(name="Central Air")
    central_air_cell.region = -inner_air_outer
    central_air_cell.fill = mat_air

    inner_wrapper_cell = openmc.Cell(name="Inner Wrapper")
    inner_wrapper_cell.region = +inner_air_outer & -inner_wrapper_outer
    inner_wrapper_cell.fill = mat_inconel

    inner_coolant_cell = openmc.Cell(name="Inner Coolant")
    inner_coolant_cell.region = +inner_wrapper_outer & -inner_coolant_outer
    inner_coolant_cell.fill = mat_water

    inner_shield_cell = openmc.Cell(name="Inner Shield")
    inner_shield_cell.region = +inner_coolant_outer & -inner_shield_outer
    inner_shield_cell.fill = mat_steel

    inner_chamber_wall_cell = openmc.Cell(name="Inner Chamber Wall")
    inner_chamber_wall_cell.region = +inner_shield_outer & -inner_chamber_outer
    inner_chamber_wall_cell.fill = mat_inconel

    plasma_cell = openmc.Cell(name="Plasma")
    plasma_cell.region = +inner_chamber_outer & -outer_chamber_inner
    plasma_cell.fill = None 

    outer_chamber_wall_cell = openmc.Cell(name="Outer Chamber Wall")
    outer_chamber_wall_cell.region = +outer_chamber_inner & -outer_shield_inner
    outer_chamber_wall_cell.fill = mat_inconel

    outer_shield_cell = openmc.Cell(name="Outer Shield")
    outer_shield_cell.region = +outer_shield_inner & -outer_coolant_inner
    outer_shield_cell.fill = mat_steel

    outer_coolant_cell = openmc.Cell(name="Outer Coolant")
    outer_coolant_cell.region = +outer_coolant_inner & -outer_wrapper_inner
    outer_coolant_cell.fill = mat_water

    outer_wrapper_cell = openmc.Cell(name="Outer Wrapper")
    outer_wrapper_cell.region = +outer_wrapper_inner & -outer_air_inner
    outer_wrapper_cell.fill = mat_inconel

    air_cell = openmc.Cell(name="Air")
    air_cell.region = +outer_air_inner & -room_liner_inner
    air_cell.fill = mat_air

    concrete_liner_cell = openmc.Cell(name="Concrete Liner")
    concrete_liner_cell.region = +room_liner_inner & -room_liner_outer
    concrete_liner_cell.fill = borated_concrete

    concrete_wall_cell = openmc.Cell(name="Concrete Wall")
    concrete_wall_cell.region = +room_liner_outer & +x_min & -x_max & +y_min & -y_max
    concrete_wall_cell.fill = mat_concrete

    cells = [central_air_cell, inner_wrapper_cell, inner_coolant_cell, inner_shield_cell,
            inner_chamber_wall_cell, plasma_cell, outer_chamber_wall_cell, outer_shield_cell,
            outer_coolant_cell, outer_wrapper_cell, air_cell,
            concrete_liner_cell, concrete_wall_cell]

    root_universe = openmc.Universe(cells=cells)
    geometry = openmc.Geometry(root_universe)

    source = openmc.IndependentSource()
    source.space = openmc.stats.Box(
        lower_left=[-R0_plasma, -R0_plasma, -R0_plasma], upper_right=[R0_plasma, R0_plasma, R0_plasma], only_fissionable=False
    )
    constraints = {}
    constraints['domains'] = [plasma_cell]
    source.constraints = constraints 
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([2449632.3277176125], [1.0])

    settings = openmc.Settings()
    settings.source = source
    settings.batches = 200
    settings.particles = 20000
    settings.run_mode = 'fixed source'

    extent_xy = 2.0 * ro
    n_xy = max(1, int(np.ceil(extent_xy / mesh_cell_size_cm)))
    mesh = openmc.RegularMesh()
    mesh.lower_left  = [-ro, -ro]
    mesh.upper_right = [ ro,  ro]
    mesh.dimension   = [n_xy, n_xy]
    dim = int(mesh.dimension[0])

    mesh_tally = openmc.Tally(name="mesh_flux")
    mesh_tally.filters = [openmc.MeshFilter(mesh)]
    mesh_tally.scores  = ['flux']
    tallies = openmc.Tallies([mesh_tally])

    model = openmc.Model(geometry=geometry, materials=mats, settings=settings, tallies=tallies)

    random_ray_model = copy.deepcopy(model)

    random_ray_model.convert_to_multigroup(method='stochastic_slab',
                                        nparticles=10000,
                                        groups=random_ray_groups,
                                        correction=MGXS_correction,
    )

    random_ray_model.convert_to_random_ray()

    random_ray_model.settings.random_ray['source_region_meshes'] = [(mesh, [random_ray_model.geometry.root_universe])]

    random_ray_model.settings.particles = 50000
    random_ray_model.settings.batches = 200
    random_ray_model.settings.inactive = 100
    random_ray_model.settings.random_ray['distance_inactive'] = 4000
    random_ray_model.settings.random_ray['distance_active'] = 20000
    random_ray_model.settings.random_ray['ray_source'] = openmc.IndependentSource(space=openmc.stats.Box(
        lower_left=[-ro, -ro, -ro], upper_right=[ro, ro, ro], only_fissionable=False
    ))
    random_ray_model.settings.random_ray['source_shape'] = 'flat'
    random_ray_model.settings.random_ray['sample_method'] = 'halton'
    random_ray_model.settings.random_ray['volume_estimator'] = volume_estimator

    wwg = openmc.WeightWindowGenerator(
        method='fw_cadis', mesh=mesh, energy_bounds=list(weight_window_edges), max_realizations=random_ray_model.settings.batches,
    )

    random_ray_model.settings.weight_window_generators = wwg

    random_ray_model.run()

    openmc.hdf5_to_wws('weight_windows.h5')

    model.settings.weight_window_checkpoints = {'collision': True, 'surface': True}
    model.settings.survival_biasing = False
    model.settings.weight_windows = openmc.hdf5_to_wws('weight_windows.h5')

    model.settings.batches = 100
    model.settings.particles = 20000
    model.settings.weight_windows_on = True
    sp_with = model.run(path="mc_with_ww.xml")
    results_with_WW = summarize_JETSON_2D_statepoint(sp_with)

    model.settings.batches = 100
    model.settings.particles = 200000
    model.settings.weight_windows_on = False
    sp_no = model.run(path="mc_no_ww.xml")
    results_no_WW = summarize_JETSON_2D_statepoint(sp_no)

    os.chdir(orig_dir)

    return results_with_WW, results_no_WW

if __name__ == '__main__':
    run_JETSON_2D()