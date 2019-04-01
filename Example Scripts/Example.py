import pyzones as pz
import numpy as np

# Create a soundfield of the correct dimensions, Soundfield(xy, width, height)
# In this instance making 0, 0 the centre of the soundfield. Rectangles can also be defined from the "bottom left"
soundfield = pz.Soundfield([0, 0], 4, 4, coordinate_pos="centre")

# Create the zones, Zone(xy, radius)
# setting the colour of the zones is optional, this only affects the visualisation at the end. The zones must be added
# to the soundfield to make them appear in the graph
bright_zone = pz.Zone([-0.5, 0], 0.11, colour=(0.5, 0.5, 0.5))
dark_zone = pz.Zone([0.5, 0], 0.11, colour=(0.2, 0.2, 0.2))
soundfield.add_zones([bright_zone, dark_zone])

# Create the loudspeaker array
num_ls = 60
# Create an array of loudspeakers, each of which is "looking at" the centre of the room in the visualisation
ls_array = pz.LoudspeakerArray([pz.Loudspeaker(look_at=[0, 0]) for _ in range(num_ls)])
# Define the shape of the loudspeaker array, Circle(xy, radius). Then position the loudspeakers around the shape.
ls_array_shape = pz.Circle([0, 0], 1.680)
ls_array.position_objects(ls_array_shape.get_perimeter(0, 0, num_ls))
# Loudspeakers then added to soundfield to ensure they appear in the visualisation
soundfield.add_sound_objects(ls_array)

# Create the different mic arrays used to set up and evaluate the system. Microphone(zone, purpose)
# The options are bright/dark/either and setup/evaluation/either respectively. In this instance all microphones are used
# to evaluate and set up.
# This creates 4 circular arrays of 24 microphones. Two in each zone, the first around the perimeter of the zone.
# The second sharing the central point of the zone but with a radius of 0.085m
num_mics_per_circle = 24
bright_setup = pz.MicrophoneArray([pz.Microphone("bright", "either") for _ in range(num_mics_per_circle)])
bright_setup.position_objects(bright_zone.get_perimeter(0, 0, num_mics_per_circle))
bright_eval = pz.MicrophoneArray([pz.Microphone("bright", "either") for _ in range(num_mics_per_circle)])
bright_eval.position_objects(bright_zone.get_circular_points(0, 0, num_mics_per_circle, 0.085))
dark_setup = pz.MicrophoneArray([pz.Microphone("dark", "either") for _ in range(num_mics_per_circle)])
dark_setup.position_objects(dark_zone.get_perimeter(0, 0, num_mics_per_circle))
dark_eval = pz.MicrophoneArray([pz.Microphone("dark", "either") for _ in range(num_mics_per_circle)])
dark_eval.position_objects(dark_zone.get_circular_points(0, 0, num_mics_per_circle, 0.085))

# These mic arrays are then added together to be used as a whole.
mic_array = bright_setup + bright_eval + dark_setup + dark_eval

# Define the frequency range of the tests - 100 to 8000 with a step of 50
frequencies = np.arange(100, 8050, 50)

# Create your simulation, optional arguments are available to set constants
sim = pz.Simulation(frequencies, ls_array, mic_array)

# Create the steering vectors for planarity control and planarity evaluation. This will take a long time.
# To save time in future simulations these can be saved and loaded - see SteeringVector documentation.
eval_str_vec = pz.SteeringVector(sim, "evaluation")
setup_str_vec = pz.SteeringVector(sim, "setup")

steering_vecs = [eval_str_vec, setup_str_vec]
sim.add_steering_vectors(steering_vecs)

# Choose the method used - ACC (acoustic contrast control), BC (Brightness Control), PC (Planarity Control),
# PM (Pressure Matching). This script loops through all of the available options.
methods = ['BC', 'ACC', 'PC', 'PM']
vis_tfs = None
grid = None
for method in methods:
    # Run your simulation by calculating the filter weights and subsequently evaluate these by calculating metrics
    sim.calculate_filter_weights(method=method)
    metrics = sim.calculate_metrics(contrast=True, effort=True, planarity=True)
    print(method)
    metrics.print(contrast=True, effort=True, planarity=True)
    metrics.output_csv("results_%s.csv" % method, overwrite=True, contrast=True, effort=True, planarity=True)
    metrics.plot("%s_effort_plot" % method, "%s_effort_plot.png" % method, "effort")

    # Create the soundfield visualisation for the most recently calculated filter weights and choose the frequency at
    # which is it visualised
    vis_frequency = 500
    vis_tfs, grid = soundfield.visualise(sim, "%s - %d.png" % (method, vis_frequency),
                                         frequency=vis_frequency, transfer_functions=vis_tfs, grid=grid)

