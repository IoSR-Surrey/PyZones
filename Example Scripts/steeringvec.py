import pyzones as pz
import numpy as np

# Create a soundfield of the correct dimensions, Soundfield(xy, width, height)
# In this instance making 0, 0 the centre of the soundfield. Rectangles can also be defined from the "bottom left"
soundfield = pz.Soundfield([0, 0], 4, 4, coordinate_pos="centre")

# Create the zones, Zone(xy, radius)
# setting the colour of the zones is optional, this only affects the visualisation at the end. The zones must be added
# to the soundfield to make them appear in the graph
bright_zone = pz.Zone([-0.5, 0], 0.11, colour=[0.5, 0.5, 0.5])
dark_zone = pz.Zone([0.5, 0], 0.11, colour=[0.2, 0.2, 0.2])
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
# The options are bright/dark/either and setup/evaluation/either respectively.
num_mics_per_block = 101
positions = np.zeros((101, 2), float)
column_length = [5, 9, 9, 11, 11, 11, 11, 11, 9, 9, 5]

bright_eval = pz.MicrophoneArray([pz.Microphone("bright", "evaluation") for _ in range(num_mics_per_block)])
pos_index = 0
for i in range(len(column_length)):
    start = -0.02 * ((column_length[i] - 1) / 2)
    for j in range(column_length[i]):
        positions[pos_index][0] = -0.6 + 0.02 * i
        positions[pos_index][1] = start + 0.02 * j
        pos_index += 1
bright_eval.position_objects(positions)

dark_eval = pz.MicrophoneArray([pz.Microphone("dark", "evaluation") for _ in range(num_mics_per_block)])
pos_index = 0
for i in range(len(column_length)):
    start = -0.02 * ((column_length[i] - 1) / 2)
    for j in range(column_length[i]):
        positions[pos_index][0] = 0.4 + 0.02 * i
        positions[pos_index][1] = start + 0.02 * j
        pos_index += 1
dark_eval.position_objects(positions)

num_mics_per_circle = 24
bright_setup = pz.MicrophoneArray([pz.Microphone("bright", "setup") for _ in range(num_mics_per_circle * 2)])
a = bright_zone.get_perimeter(0, 0, num_mics_per_circle)
b = bright_zone.get_circular_points(0, 0, num_mics_per_circle, 0.085)
positions = np.concatenate((a, b))
bright_setup.position_objects(positions)

dark_setup = pz.MicrophoneArray([pz.Microphone("dark", "setup") for _ in range(num_mics_per_circle * 2)])
a = dark_zone.get_perimeter(0, 0, num_mics_per_circle)
b = dark_zone.get_circular_points(0, 0, num_mics_per_circle, 0.085)
positions = np.concatenate((a, b))
dark_setup.position_objects(positions)

# These mic arrays are then added together to be used as a whole.
mic_array = bright_setup + bright_eval + dark_setup + dark_eval

# Define the frequency range of the tests - 100 to 8000 with a step of 50
frequencies = np.arange(50, 8000, 50)

# Create your simulation, optional arguments are available to set constants
sim = pz.Simulation(frequencies, ls_array, mic_array)

# eval_str_vec = pz.SteeringVector(sim, "evaluation", file_path="/Users/iosr/Documents/PyZones/bright_eval_str_vec.csv")
# setup_str_vec = pz.SteeringVector(sim, "setup", file_path="/Users/iosr/Documents/PyZones/bright_setup_str_vec.csv")
#
eval_str_vec = pz.SteeringVector(sim, "evaluation")
setup_str_vec = pz.SteeringVector(sim, "setup")

steering_vecs = [eval_str_vec, setup_str_vec]
sim.add_steering_vectors(steering_vecs)

methods = ['BC', 'ACC', 'PC', 'PM']
vis_tfs = None
grid = None
for method in methods:

    # Run your simulation by calculating the filter weights and subsequently evaluate these by calculating metrics
    sim.calculate_filter_weights(method=method)
    metrics = sim.calculate_metrics(contrast=True, effort=True, planarity=True)
    sim.calculate_metrics(contrast=True, effort=True)
    print(method)
    metrics.print(contrast=True, effort=True, planarity=True)
    metrics.output_csv("/Users/iosr/Desktop/test_%s.csv" % method, overwrite=True, contrast=True, effort=True)
    metrics.plot("test", "test", "effort")

    # Create the soundfield visualisation for the most recently calculated filter weights and choose the frequency at
    # which is it visualised
    vis_frequency = 500
    vis_tfs, grid = soundfield.visualise(sim, "%s - %d.png" % (method, vis_frequency), frequency=vis_frequency, transfer_functions=vis_tfs, grid=grid)


