from abc import abstractmethod
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import csv

# ********* Geometric *********


class Shape:
    """
    An abstract class for geometric shapes defining some key methods required

    """

    @abstractmethod
    def get_perimeter(self, start, end, num_points):
        """
        Create a list of points between the user defined start and end positions on the perimeter of the shape

        :param start: Position at which the list of points should begin
        :type start: float
        :param end: Position at which the list of points should end
        :type end: float
        :param num_points: Number of points
        :type num_points: int
        :return: A list of points (x,y) evenly spaced on the perimeter of the shape between the start and end positions
        :rtype: numpy.ndarray

        """
        pass

    @abstractmethod
    def get_grid(self, spacing):
        """
        Create a grid of points spaced uniformly across the shape

        :param spacing: Spacing between points in the grid
        :type spacing: float
        :return: A list of points (x,y) uniformly space across the shape
        :rtype: numpy.ndarray

        """
        pass

    @abstractmethod
    def is_point_inside(self, point):
        """
        Check whether or not a point is inside the shape

        :param point: list/tuple of the coordinates (x, y) of a point
        :type point: list
        :return: A bool stating whether or not the point is within the shape
        :rtype: bool

        """
        pass


class Circle(Shape):
    """
    A geometric class for a circle

    Attributes
    ----------
    centre : list
        A list of coordinates (x, y) describing the centre of the circle
    radius : float
        The radius of the circle

    """

    def __init__(self, centre, radius):
        """
        Creates a circle

        :param centre: The coordinates (x,y) of centre of the circle
        :type centre: list
        :param radius: The radius of the circle
        :type radius: float
        :rtype: Circle

        """

        self._centre = centre
        self._radius = radius

    @property
    def centre(self):
        """
        A list of coordinates (x, y) describing the centre of the circle

        :return: (x, y) of the centre of the circle
        :rtype: list

        """

        return self._centre

    @property
    def radius(self):
        """
        The radius of the circle

        :return: The radius
        :rtype: float

        """

        return self._radius

    def get_circular_points(self, start_angle, end_angle, num_points, radius, decimal_places=None):
        """
        Create a list of points between the user defined start and end angles (in degrees) on the perimeter of a new circle sharing
        the centre point of this circle with a different radius

        :param start_angle: Position at which the list of points should begin
        :type start_angle: float
        :param end_angle: Position at which the list of points should end
        :type end_angle: float
        :param num_points: Number of points
        :type num_points: int
        :param radius: Radius of the circle on which the points are placed
        :type radius: float
        :param decimal_places: Number of decimal places the coordinates are returned with - None: there is no rounding
        :type decimal_places: int
        :return: An array of points (x,y) evenly spaced on the perimeter of the new circle between the start and end angles
        :rtype: numpy.ndarray

        """

        points = np.zeros((num_points, 2), float)

        full_angle = 180 - abs(abs(end_angle - start_angle) - 180)
        if full_angle == 0:
            full_angle = 360
        delta_angle = full_angle / num_points

        for i in range(num_points):
            points[i][0] = self._centre[0] + np.cos(np.radians(90 + start_angle + delta_angle * i)) * radius
            points[i][1] = self._centre[1] + np.sin(np.radians(90 + start_angle + delta_angle * i)) * radius
        if decimal_places is not None:
            return np.array(np.around(points, decimal_places))
        else:
            return np.array(points)

    def get_perimeter(self, start_angle, end_angle, num_points, decimal_places=None):
        """
        Create a list of points between the user defined start and end angles on the perimeter of the circle

        :param start_angle: Position at which the list of points should begin
        :type start_angle: float
        :param end_angle: Position at which the list of points should end
        :type end_angle: float
        :param num_points: Number of points
        :type num_points: int
        :param decimal_places: Number of decimal places the coordinates are returned with - None: there is no rounding
        :type decimal_places: int
        :return: A list of points (x,y) evenly spaced on the perimeter of the shape between the start and end angles
        :rtype: numpy.ndarray

        """

        return np.array(self.get_circular_points(start_angle, end_angle, num_points, self._radius, decimal_places))

    def get_grid(self, spacing, alpha=2):
        """
        Create a grid of points spaced uniformly across the circle using the sunflower seed arrangement algorithm

        :param spacing: Approximate spacing between points in the grid
        :type spacing: float
        :param alpha: Determines the evenness of the boundary - 0 is jagged, 2 is smooth. Above 2 is not recommended
        :type alpha: float
        :return: A list of points (x,y) uniformly spaced across the circle
        :rtype: numpy.ndarray

        """

        # Algorithm is found at the stack overflow thread linked below:
        # https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle

        # Calculates the number of points (n) from the spacing
        area = np.pi * self._radius**2
        n = int(area / spacing**2)
        points = np.zeros((n, 2), float)

        b = int(alpha * np.sqrt(n)) # number of boundary points
        golden_ratio = (np.sqrt(5) + 1) / 2

        for point in range(1, n + 1):
            if point > n - b:
                r = 1
            else:
                r = np.sqrt(point - 1 / 2) / np.sqrt(n - (b + 1) / 2)

            theta = 2 * np.pi * point / golden_ratio**2

            points[point - 1][0] = self._centre[0] + r*np.cos(theta) * self._radius
            points[point - 1][1] = self._centre[1] + r*np.sin(theta) * self._radius

        return np.array(points)

    def is_point_inside(self, point):
        """
        Check whether or not a point is inside the circle

        :param point: List/tuple of the coordinates (x, y) of a point
        :type point: list
        :return: A bool stating whether or not the point is within the circle
        :rtype: bool

        """

        # checks if the distance from the centre of the circle to the point, d, is less than or equal to the radius
        d = np.sqrt((point[0] - self.centre[0])**2 + (point[1] - self.centre[1])**2)
        return d <= self.radius


class Rectangle(Shape):
    """
    A geometric class for a rectangle

    Attributes
    ----------
    coordinate : list
        A list of coordinates (x, y) describing the centre or bottom left of the rectangle
    width : float
        The width of the rectangle
    height : float
        The height of the rectangle
    coordinate_pos : str
        Describes the position of the coordinate parameter - either "centre" or "bottom left"

    """

    def __init__(self, coordinate, width, height, coordinate_pos="bottom left"):
        """
        Creates a rectangle

        :param coordinate: A list of coordinates (x, y) describing the centre or bottom left of the rectangle
        :type coordinate: list
        :param width: The width of the rectangle
        :type width: float
        :param height: The height of the rectangle
        :type height: float
        :param coordinate_pos: Description of the position of the coordinate - "centre" or "bottom left" of the rectangle
        :type coordinate_pos: str
        :rtype: Rectangle

        """

        if coordinate_pos == 'centre':
            self._xy = [coordinate[0] - width / 2, coordinate[1] - height / 2]
        elif coordinate_pos == 'bottom left':
            self._xy = coordinate
        else:
            print("coordinate_pos must be in \"centre\" or \"bottom left\"")
            quit(1)
        self._width = width
        self._height = height

    @property
    def xy(self):
        """
        A list of coordinates (x, y) describing the bottom left of the rectangle

        :return: (x, y) of the bottom left of the rectangle
        :rtype: list

        """

        return self._xy

    @property
    def width(self):
        """
        The width of the rectangle

        :return: The width
        :rtype: float

        """

        return self._width

    @property
    def height(self):
        """
        The height of the rectangle

        :return: The height
        :rtype: float

        """

        return self._height

    def get_perimeter(self, start_point, end_point, num_points):
        pass

    def get_grid(self, spacing):
        """
        Create a grid of points spaced uniformly across the rectangle

        :param spacing: Approximate spacing between points in the grid
        :type spacing: float
        :return: A list of points (x,y) uniformly spaced across the rectangle
        :rtype: numpy.ndarray

        """

        num_x = int(np.floor(self._width / spacing)) + 1
        num_y = int(np.floor(self._height / spacing)) + 1
        num_points = int(num_x * num_y)
        points = np.zeros((num_points, 2), float)
        for x in range(num_x):
            for y in range(num_y):
                points[y * num_x + x][0] = self._xy[0] + x * spacing
                points[y * num_x + x][1] = self._xy[1] + y * spacing
        return np.array(points)

    def is_point_inside(self, point):
        """
        Check whether or not a point is inside the rectangle

        :param point: list/tuple of the coordinates (x, y) of a point
        :type point: tuple
        :return: A bool stating whether or not the point is within the rectangle
        :rtype: bool

        """

        # checks that the x and y distance between the point and the bottom_left of the rectangle is less than the
        # width and height
        dx = abs(point[0] - self._xy[0]) + abs(self._xy[0] + self._width - point[0])
        dy = abs(point[1] - self._xy[1]) + abs(self._xy[1] + self._height - point[1])
        return dy <= self._height and dx <= self._width

# ********* PyZones specific setup classes *********


class Zone(Circle):
    """
    A sound zone to be used in setup of the soundfield's geometry

    Attributes
    ----------
    centre : list
        A list of coordinates (x, y) describing the centre of the circle
    radius : float
        The radius of the circle
    colour : list
        A list of float values (r, g, b)

    """

    def __init__(self, centre, radius, colour=None):
        """
        Creates a sound zone

        :param centre: A list of coordinates (x, y) describing the centre of the circle
        :type centre: list
        :param radius: The radius of the circle
        :type radius: float
        :param colour: A list of float values (r, g, b) - None results in black (0, 0, 0)
        :type colour: list
        :rtype: Zone

        """

        if colour is None:
            self._colour = [0, 0, 0]
        else:
            self._colour = colour

        Circle.__init__(self, centre, radius)

    @property
    def colour(self):
        """
        A list of float values (r, g, b)

        :return: A list of float values (r, g, b)
        :rtype: list

        """

        return self._colour


class Soundfield(Rectangle):
    """
    The soundfield being used in the simulation. Can be thought of as the room, however no room reflections are modelled

    Attributes
    ----------
    _zones : list
        A list of the zones used in the simulation.
    _fig : float
        The figure from the matplotlib.pyplot
    _axes : float
        The axes from the matplotlib.pyplot

    """

    def __init__(self, coordinate, width, height, coordinate_pos="bottom left"):
        """
        Creates a soundfield to be used for simulations. This class is exclusively for the graphics and visualisations

        :param coordinate: A list of coordinates (x, y) describing the centre or bottom left of the rectangle
        :type coordinate: list
        :param width: The width of the rectangle
        :type width: float
        :param height: The height of the rectangle
        :type height: float
        :param coordinate_pos: The position of the coordinate - "centre" or "bottom left" of the rectangle
        :type coordinate_pos: str
        :rtype: Soundfield

        """

        Rectangle.__init__(self, coordinate, width, height, coordinate_pos=coordinate_pos)
        self._zones = []

        self._fig = plt.figure(figsize=(6, 6), dpi=300)
        self._axes = self._fig.add_subplot(111)
        self._axes.set_xlim([self.xy[0], self.xy[0] + width])
        self._axes.set_ylim([self.xy[1], self.xy[1] + height])
        self._cax = self._fig.add_axes([0.125, 0.94, 0.775, 0.04])

    def add_zones(self, zones):
        """
        Add the sound zone(s) to the soundfield such that they can be seen in the visualisations of the soundfield

        :param zones: The zone(s) to be added to the soundfield
        :type zones: list[Zone]

        """

        if type(zones) is not list:
            zones = [zones]

        for zone in zones:
            circle = plt.Circle(zone.centre, zone.radius, fill=False)
            circle.set_edgecolor(zone.colour)
            self._axes.add_patch(circle)
            self._zones.append(zone)

    def add_sound_objects(self, *args):
        """
        Add the sound objects to the soundfield such that they can be seen in the visualisations of the soundfield

        :param args: a single Microphone/Loudspeaker or a MicrophoneArray/LoudspeakerArray

        """

        def add_ls(ls):
                centre = ls.position
                x = centre[0] - (ls.width / 2)
                y = centre[1] - (ls.height / 2)
                angle = 0

                # change the orientation of the loudspeaker such that it's looking at a point (purely aesthetic)
                if ls.look_at is not None:
                    x_dif = ls.look_at[0] - centre[0]
                    y_dif = ls.look_at[1] - centre[1]

                    if x_dif == 0:
                        angle = 0
                    elif y_dif == 0:
                        angle = np.pi / 2
                    elif x_dif > 0:
                        angle = np.arctan(y_dif / x_dif) - np.pi / 2
                    else:
                        angle = np.arctan(y_dif / x_dif) + np.pi / 2


                    new_x = (x - centre[0]) * np.cos(angle) - (y - centre[1]) * np.sin(angle) + centre[0]
                    new_y = (x - centre[0]) * np.sin(angle) + (y - centre[1]) * np.cos(angle) + centre[1]
                    x = new_x
                    y = new_y


                rect = plt.Rectangle((x, y), ls.width, ls.height, angle=np.rad2deg(angle), fill=False)
                rect.set_edgecolor(ls.colour)
                self._axes.add_patch(rect)

        def add_mic(mic):
                circle = plt.Circle(mic.position, mic.radius, fill=False)
                circle.set_edgecolor(mic.colour)
                self._axes.add_patch(circle)

        for s_object in args:
            if isinstance(s_object, Loudspeaker):
                add_ls(s_object)
            elif isinstance(s_object, LoudspeakerArray):
                for item in s_object:
                    add_ls(item)
            elif isinstance(s_object, Microphone):
                add_mic(s_object)
            elif isinstance(s_object, MicrophoneArray):
                for item in s_object:
                    add_mic(item)
            else:
                "Please input a Microphone/Loudspeaker or MicrophoneArray/LoudspeakerArray to add."
                return

    def clear_graphs(self):
        """
        Clear the SoundObjects and Zones from the visualisations

        """

        self._axes.clear()

    def plot_geometry(self, graph_name):
        """
        Plot the geometry of the soundfield with any Zones or SoundObjects added

        :param graph_name: The name and file location of the graph
        :type graph_name: str

        """

        self._axes.plot()
        self._fig.savefig(graph_name)

    def visualise(self, sim, graph_name, frequency=500, sf_spacing=0.1, zone_spacing=0.05, zone_alpha=2, transfer_functions=None, grid=None):
        """
        Create a visualisation of the Soundfield at the given frequency. The frequency chosen must have been present in
        the simulation provided. Transfer functions and visualisation micr positions can be provided to prevent them
        being calculated more than once. Should the same frequency, loudspeakers and visualisation microphone positions
        be kept the same, the returned transfer functions and microphone positions can be used again. The filter weights
        used will be those most recently calculated in the simulation.

        :param sim: Simulation for which the visualation is made - contains the filter weights.
        :type sim: Simulation
        :param graph_name: The name and file location of the graph
        :type graph_name: str
        :param frequency: Frequency at which the visualisation should be made - must have been present in the Simulation
        :type frequency: int
        :param sf_spacing: The spacing between the microphones in the grid across the soundfield in metres
        :type sf_spacing: float
        :param zone_spacing: The spacing between the microphones in the grid in the zone in metres
        :type zone_spacing: float
        :param zone_alpha: Determines the evenness of the boundary - 0 is jagged, 2 is smooth. Above 2 is not recommended
        :type zone_alpha: float
        :param transfer_functions: ndarray of transfer functions shape (microphones, loudspeakers). None - calculated
        :type transfer_functions: numpy.ndarray
        :param grid: Grid of mic positions (x, y) matching up with the provided transfer function. None - calculated
        :type grid: list
        :return: The grid and transfer functions used in the visualisation to prevent their unnecessary recalculation.
        :rtype: numpy.ndarray, list

        """

        # create the grid of microphones for visualisation
        if grid is None:
            points = self.get_grid(sf_spacing)
            for zone in self._zones:
                points = np.concatenate((points, zone.get_grid(zone_spacing, alpha=zone_alpha)), 0)
        else:
            points = grid

        # create the transfer functions for the vis mics
        if transfer_functions is None:
            vis_mics = MicrophoneArray([Microphone(position=points[i]) for i in range(len(points))])
            tfs = sim.calculate_transfer_functions("microphones", mic_array=vis_mics, frequency=frequency)
        else:
            tfs = transfer_functions

        # find the simulation frequency index of the visualisation frequency
        nonzero_array = np.nonzero(sim.frequencies == frequency)
        if len(nonzero_array[0]) == 0:
            print("Please visualise a frequency for which filter weights have been calculated")
            return
        freq_index = nonzero_array[0][0]

        # use the most recently calculated filter weights corresponding to the visualisation frequency to calc pressure
        q_matrix = np.array(sim.ls_array.get_q(frequency_index=freq_index))
        p = (tfs[0] @ q_matrix[:, None]).flatten()
        p = np.abs(p)
        p = convert_to_db(p)

        # plot the pressure
        step = 0.01
        xi = np.arange(self._xy[0], self._xy[0] + self._width + step, step)
        yi = np.arange(self._xy[1], self._xy[1] + self._height + step, step)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata(points, p, (xi, yi), method='cubic', fill_value=1)

        self._axes.contourf(xi, yi, zi, np.arange(0, 1.01, 0.01))
        colours = self._axes.pcolormesh(xi, yi, zi, vmin=0, vmax=90)

        self._fig.colorbar(colours, cax=self._cax, orientation='horizontal')

        # self._axes.pcolormesh(xi, yi, zi, vmin=0, vmax=90) # remove once colour bar added

        # self._axes.plot(*zip(*points), marker=',', color='r', ls='') SHOWS MIC POSITIONS
        self._fig.savefig(graph_name, dpi=300)

        return tfs, points


class SoundObject:
    """
    A sound object to be used in simulations of sound zones

    Attributes
    ----------
    colour : list
        A list of float values (r, g, b)
    position : list
        A list of coordinates (x, y) describing the centre of the sound object

    """
    def __init__(self, position=None, colour=None):
        """
        Creates a sound object for use in sound zone simulations

        :param position: A list of coordinates (x, y) describing the position of the sound object
        :type position: list
        :param colour: A list of float values (r, g, b) - None results in black (0, 0, 0)
        :type colour: list

        """

        if position is None:
            self._position = [0, 0]
        else:
            self._position = position
        if colour is None:
            self._colour = [0, 0, 0]
        else:
            self._colour = colour

    @property
    def colour(self):
        """
        A list of float values (r, g, b)

        :return: A list of float values (r, g, b)
        :rtype: list

        """

        return self._colour

    @property
    def position(self):
        """
        A list of coordinates (x, y) describing the position of the sound object

        :return: (x, y) of the bottom left of the rectangle
        :rtype: list

        """

        return self._position

    @position.setter
    def position(self, val):
        """
        Set A list of coordinates (x, y) describing the position of the sound object

        """

        self._position[0] = val[0]
        self._position[1] = val[1]


class Microphone(SoundObject):
    """
    A microphone to be used in simulations of sound zones. Inherits from the sound object class.

    Class Attributes
    ----------------
    _radius : static float
        The radius of circles used to represent the microphones when rendered in the soundfield

    Attributes
    ----------
    colour : list
        A list of float values (r, g, b)
    position : list
        A list of coordinates (x, y) describing the position of the microphone
    zone : str
        The zone in which this microphone is situated, "bright", "dark" or "either"
    purpose : str
        The purpose of the microphone, "setup", "evaluation" or "either
    _pressure : list
        The pressure for each frequency at the microphone most recently calculated and set

    """

    _radius = 0.001

    @property
    def radius(self):
        """
        The radius of circles used to represent the microphones when rendered in the soundfield

        :return: The radius
        :rtype: float

        """

        return type(self)._radius

    @radius.setter
    def radius(self, val):
        """
        Set the radius of circles used to represent the microphones when rendered in the soundfield

        :param val: Value to be set as radius
        :type val: float

        """

        _radius = val

    def __init__(self, zone="none", purpose="none", position=None, colour=None):
        """
        Creates a microphone to be used in sound zone simulations

        :param zone: The zone in which this microphone is situated, "bright", "dark" or "either"
        :type zone: str
        :param purpose: The purpose of the microphone, "setup", "evaluation" or "either"
        :type purpose: str
        :param position: A list of coordinates (x, y) describing the centre of the sound object
        :type position: list
        :param colour: A list of float values (r, g, b) - None results in black (0, 0, 0)
        :type colour: list

        """

        SoundObject.__init__(self, position, colour)
        self._zone = zone
        self._purpose = purpose
        self._pressure = []

    @property
    def zone(self):
        """
        The zone in which this microphone is situated, "bright" or "dark"

        :return: The zone
        :rtype: str

        """

        return self._zone

    @property
    def purpose(self):
        """
        The purpose of the microphone, "setup" or "evaluation"

        :return: The purpose
        :rtype: str

        """

        return self._purpose

    @property
    def pressure(self):
        """
        The pressure for each frequency at the microphone most recently calculated and set

        :return: The pressure
        :rtype: list

        """

        return self._pressure

    @pressure.setter
    def pressure(self, list):
        """
        The pressure for each frequency at the microphone most recently calculated and set

        :param list: List of pressures at each frequency
        :type list: list

        """

        self._pressure = list


class Loudspeaker(SoundObject):
    """
    A loudspeaker to be used as a source in simulations of sound zones. Inherits from the sound object class.

    Class Attributes
    ----------------
    _width : static float
        The width of rectangles used to represent loudspeakers when rendered in the soundfield
    _height : static float
        The height of rectangles used to represent loudspeakers when rendered in the soundfield

    Attributes
    ----------
    colour : list
        A list of float values (r, g, b)
    position : list
        A list of coordinates (x, y) describing the position of the loudspeaker
    look_at : list
        A list of coordinates (x, y) describing the position the loudspeaker faces
    q : list
        The filter weight at each frequency most recently calculated and set

    """

    _width = 0.08
    _height = 0.1

    @property
    def width(self):
        """
        The width of rectangles used to represent loudspeakers when rendered in the soundfield

        :return: The width
        :rtype: float

        """

        return type(self)._width

    @property
    def height(self):
        """
        The height of rectangles used to represent loudspeakers when rendered in the soundfield

        :return: The height
        :rtype: float

        """

        return type(self)._height

    def __init__(self, position=None, colour=None, look_at=None):
        """
        Creates a Loudspeaker to be used as a source in sound zone simulations

        :param position: A list of coordinates (x, y) describing the position of the loudspeaker
        :type position: list
        :param colour: A list of float values (r, g, b)
        :type colour: list
        :param look_at: A list of coordinates (x, y) describing the position the loudspeaker faces
        :type look_at: list

        """

        SoundObject.__init__(self, position, colour)
        self._look_at = look_at
        self.q = []

    @property
    def look_at(self):
        """
        A list of coordinates (x, y) describing the position the loudspeaker faces

        :return: a list of coordinates (x, y)
        :rtype: list

        """

        return self._look_at


class SoundObjectArray(list):
    """
    A container class for sound objects

    """

    def __init__(self, *args):
        """
        Creates an array of sound objects

        """

        list.__init__(self, *args)

    def position_objects(self, positions):
        """
        Position the sound objects

        :param positions: A list of positions the same length as the number of objects
        :type positions: numpy.ndarray

        """

        for i in range(len(positions)):
            self[i].position = positions[i]

    def get_object_positions(self):
        """
        Returns a list of the positions of the sound objects

        :return: list of positions
        :rtype: list

        """

        return [self[i].position for i in range(len(self))]

    def __add__(self, other):
        return type(self)(list.__add__(self, other))

    def __iadd__(self, other):
        return type(self)(list.__add__(self, other))


class LoudspeakerArray(SoundObjectArray):
    """
    A container class for loudspeakers

    """

    def initialise_q(self, num_frequencies):
        """
        Initialise the filter weights of the loudspeakers in the loudspeaker array to one.

        :param num_frequencies: The number of frequencies in the simulation.
        :type num_frequencies: int

        """

        for ls in self:
            ls.q = np.ones(num_frequencies, complex)

    def set_q(self, new_q, frequency_index):
        """
        Set the filter weight values for the loudspeaker array at the frequency index

        :param new_q: The list of filter weights
        :type new_q: list
        :param frequency_index: The index at which the relevant frequency is stored in the simulation's frequency vector
        :type frequency_index: int

        """

        for i in range(len(new_q)):
            self[i].q[frequency_index] = new_q[i]

    def get_q(self, frequency_index):
        """
        Get the filter weight values for the loudspeaker array at the frequency index

        :param frequency_index: The index at which the relevant frequency is stored in the simulation's frequency vector
        :type frequency_index: int
        :return: The list of filter weights calculated for the relevant frequency
        :rtype: list

        """

        if frequency_index < 0:
            return [ls.q for ls in self]
        else:
            return [ls.q[frequency_index] for ls in self]


class MicrophoneArray(SoundObjectArray):
    """
    A container class for microphones

    """

    def initialise_pressures(self, num_frequencies):
        """
        Initialise the pressures of the microphones in the microphone array to zero.

        :param num_frequencies: The number of frequencies in the simulation.
        :type num_frequencies: int

        """

        for mic in self:
            mic.pressure = np.zeros(num_frequencies, complex)

    def set_pressures(self, new_pressures, frequency_index):
        """
        Set the pressures values for the microphone array at the frequency index

        :param new_pressures: The list of pressures
        :type new_pressures: list
        :param frequency_index: The index at which the relevant frequency is stored in the simulation's frequency vector
        :type frequency_index: int

        """

        for i in range(len(new_pressures)):
            self[i].pressure[frequency_index] = new_pressures[i]

    def get_pressures(self, frequency_index):
        """
        Returns the pressure values for the microphone array at the frequency index

        :param frequency_index: The index at which the relevant frequency is stored in the simulation's frequency vector
        :type frequency_index: int
        :return: The list of pressures for the relevant frequency
        :rtype: list

        """

        if frequency_index < 0:
            return [mic.pressure for mic in self]
        else:
            return [mic.pressure[frequency_index] for mic in self]

    def get_subset(self, zone="either", purpose="either"):
        """
        Returns a subset of microphones from the array with the required properties, "bright", "dark" or "either" zone
        and "setup", "evaluation" or "either" purpose.

        :param zone: The zone in which the subset of microphones should be positioned - "bright", "dark" or "either"
        :type zone: str
        :param purpose: The purpose of the subset of microphones - "setup", "evaluation" or "either"
        :type purpose: str
        :return: A microphone array containing microphones of the specified requirements
        :rtype: MicrophoneArray

        """

        mics = MicrophoneArray()

        for i in range(len(self)):
            if (self[i].zone in (zone, "either") or zone is "either") and \
                    (self[i].purpose in (purpose, "either") or purpose is "either"):
                mics.append(self[i])
        return mics


# ********* Simulation and Evaluation *********

class Simulation:
    """
    Where all of the calculations for the simulation happen. Using the input of the different array and zone
    geometries calculates filter weights across a range of frequencies for different methods of sound zone optimisation
     - Brightness Control, Acoustic Contrast Control, Planarity Control and Pressure Matching.

    Attributes
    ----------
    frequencies : numpy.ndarray
        vector of frequencies
    _omega : numpy.ndarray
        vector of angular frequencies
    _k : numpy.array
        vector of wave numbers
    _c : float
        the speed of sound
    _rho : float
        the density of air
    _target_spl : float
        The target SPL in dB in the bright zone
    _target_pa : float
        The target pressure in the bright zone
    ls_array : LoudspeakerArray
        The loudspeaker array used in the simulation
    mic_array : MicrophoneArray
        The microphone array used in the simulation
    _tf_array_ls : numpy.ndarray
        The transfer functions with the loudspeakers in the first axis
    _tf_array_mic : numpy.ndarray
        The transfer functions with the microphones in the first axis
    _q_ref : float
        The reference filter weight for a loudspeaker to realise the target SPL. Used in effort calculations
    _current_method : str
        A string of the current method, i.e the most recently calculated filter weights and metrics refer to this method
    steering_vectors : list
        A list of steering vectors to contain the setup and evaluation steering vectors for the bright zone
    _planarity_constants : tuple
        A tuple containing three constants for Planarity Control. (start_angle, end_angle, transition_angle)
    _pm_angle : float
        The angle of incidence for the Pressure Matching method.

    """

    def __init__(self, frequencies, ls_array, mic_array, c=343, rho=1.21, target_spl=76,
                 planarity_constants=(360, 360, 5), pm_angle=0):
        """
        Creates a Simulation object to create transfer functions and group important constants for
        simulation calculations

        :param frequencies: The list of frequencies to run the simulations over
        :type frequencies: numpy.ndarray
        :param ls_array: The loudspeaker array used
        :type ls_array: LoudspeakerArray
        :param mic_array: The microphone array used
        :type mic_array: MicrophoneArray
        :param c: The speed of the sound
        :type c: float
        :param rho: The density of air
        :type rho: float
        :param target_spl: The target SPL in the bright zone
        :type target_spl:
        :param planarity_constants: A tuple of floats containing the start_angle, end_angle and transiton zone for the
        angular window in planarity control
        :type planarity_constants: tuple
        :param pm_angle: The angle of incidence for pressure matching
        :type pm_angle: float

        """

        if isinstance(frequencies, int):
            frequencies = [frequencies]
        self.frequencies = np.array(frequencies)

        self._omega = [frequencies[i] * 2 * np.pi for i in range(len(frequencies))]
        self._k = [self._omega[i] / c for i in range(len(frequencies))]

        self._c = c
        self._rho = rho
        self._target_spl = target_spl
        self._target_pa = 0.00002 * 10 ** (target_spl/20)

        self.ls_array = ls_array
        self.ls_array.initialise_q(len(frequencies))
        self.mic_array = mic_array
        self.mic_array.initialise_pressures(len(frequencies))

        self._tf_array_ls = np.array(self.calculate_transfer_functions("loudspeakers"))
        self._tf_array_mic = np.transpose(np.array(self._tf_array_ls), (0, 2, 1))

        bright_mics, ga = self.get_tf_subset("loudspeakers", zone="bright")
        avg_pressure = 0
        for i in range(len(frequencies)):
            ref_source = ga[i][0]
            avg_pressure += np.sqrt((ref_source.conj().T @ ref_source) / len(bright_mics))
        avg_pressure /= len(frequencies)
        self._q_ref = self._target_pa / avg_pressure

        self._current_method = None

        self.steering_vectors = [None, None]
        self._planarity_constants = planarity_constants
        self._pm_angle = pm_angle

    @property
    def omega(self):
        """
        vector of angular frequencies

        :return: vector of angular frequencies
        :rtype: numpy.ndarray

        """

        return self._omega

    @property
    def c(self):
        """
        The speed of sound

        :return: the speed of sound
        :rtype: float

        """

        return self._c

    @property
    def rho(self):
        """
        The density of air

        :return: The density of air
        :rtype: float

        """

        return self._rho

    @property
    def k(self):
        """
        vector of wave numbers

        :return: vector of wave numbers
        :rtype: numpy.ndarray

        """

        return self._k

    @property
    def target_spl(self):
        """
        The target SPL in the bright zone

        :return: The target SPL in the bright zone
        :rtype: float

        """

        return self._target_spl

    def calculate_transfer_functions(self, orientation, mic_array=None, frequency=None):
        """
        Calculate transfer functions with the given orientation. If no mic_array or frequencies are provided the
        transfer functions returned are those of the mic_array and frequency initialised when creating the simulation.
        Be advised that the transfer functions for the simulation are already made when you create a simulation object.
        This should only need to be used for external calculations hence why the optional parameters are provided.

        :param orientation: String to choose the orientation of transfer functions - "microphones" or "loudspeakers"
        :type orientation: str
        :param mic_array: The MicrophoneArray to be used. Defaults to None - the simulations microphone array
        :type mic_array: MicrophoneArray
        :param frequency: The frequencies across which the transfer functions are calculated
        :type frequency: numpy.ndarray
        :return: A array of shape (freq, mics, ls) or (freq, ls, mic) depending on orientation
        :rtype: numpy.ndarray

        """

        if mic_array is None:
            mic_array = self.mic_array

        if frequency is None:
            k = self._k
        elif type(frequency) is int:
            k = [(2 * np.pi * frequency) / self.c]
        else:
            k = (2 * np.pi * frequency) / self.c

        mic_num = len(mic_array)
        ls_num = len(self.ls_array)

        r = np.zeros((ls_num, mic_num), float)
        tf_array = np.zeros((len(self.frequencies), ls_num, mic_num), complex)

        for m in range(mic_num):
            for n in range(ls_num):
                # distance from each source to each microphone
                r[n][m] = np.sqrt((mic_array[m].position[0] - self.ls_array[n].position[0]) ** 2 + \
                                  (mic_array[m].position[1] - self.ls_array[n].position[1]) ** 2)

                for i in range(len(k)):
                    if r[n][m] == 0:
                        tf_array[i][n][m] = 1
                    else:
                        tf_array[i][n][m] = (1 / (4 * np.pi * r[n][m])) * np.exp(-1j * k[i] * r[n][m])

        if orientation == "microphones":
            return np.transpose(np.array(tf_array), (0, 2, 1))
        elif orientation == "loudspeakers":
            return np.array(tf_array)

    def get_transfer_functions(self, orientation):
        """
        Returns the transfer functions of the simulations with the given orientation

        :param orientation: String to choose the orientation of transfer functions - "microphones" or "loudspeakers"
        :type orientation: str
        :return: transfer functions
        :rtype: numpy.ndarray

        """

        if orientation == "microphones":
            return self._tf_array_mic
        elif orientation == "loudspeakers":
            return self._tf_array_ls

    def get_tf_subset(self, orientation, zone="either", purpose="either"):
        """
        Returns a subset of the simulation's transfer functions with the given orientation. The subset is chosen based
        on the parameters zone and purpose which can be. "bright", "dark" or "either" and "setup", "evaluation" or
        "either" respectively. This also returns a subset MicrophoneArray of the the microphones used in the transfer
        functions.

        :param orientation: String to choose the orientation of transfer functions - "microphones" or "loudspeakers"
        :type orientation: str
        :param zone: The zone from which the subset is taken, "bright", "dark" or "either"
        :type zone: str
        :param purpose: The purpose which the subset has, "setup", "evaluation" or "either"
        :type purpose: str
        :return: MicrophoneArray and transfer functions of the subset
        :rtype: MicrophoneArray, numpy.ndarray

        """

        tfs = self.get_transfer_functions("microphones")
        tfs = tfs.transpose(1, 2, 0)
        mics = MicrophoneArray()

        tf_subset = []

        for i in range(len(self.mic_array)):
            if (self.mic_array[i].zone in (zone, "either") or zone is "either") and (self.mic_array[i].purpose in (purpose, "either") or purpose is "either"):
                tf_subset.append(tfs[i])
                mics.append(self.mic_array[i])

        if orientation is "microphones":
            return mics, np.transpose(np.array(tf_subset), (2, 0, 1))
        elif orientation is "loudspeakers":
            return mics, np.transpose(np.array(tf_subset), (2, 1, 0))

    def calculate_filter_weights(self, method='BC', beta=0.01):
        """
        Calculate filter weights for the loudspeaker array based on the transfer functions of the microphones in the mic
        array with the purpose of "bright" or "either". Four different methods are available - Brightness control 'BC',
        Acoustic Contrast Control 'ACC', Planarity Control 'PC' and Pressure Matching 'PM'. These filter weights are not
        returned but automatically assigned to the loudspeaker array.

        :param method: The string containing the method for which the filter weights should be calculated.
        :type method: str

        """

        def scale_q(q_unscaled, frequency_index):
            self.ls_array.set_q(q_unscaled, frequency_index=frequency_index)
            alpha = self._target_pa / self.get_avg_pa(i, purpose="setup", zone="bright")
            q_scaled = [Q * alpha for Q in q_unscaled]
            self.ls_array.set_q(q_scaled, frequency_index=frequency_index)

        self.ls_array.initialise_q(len(self.frequencies))
        self._current_method = method

        setup_bright, g_a = self.get_tf_subset("microphones", purpose="setup", zone="bright")
        setup_dark, g_b = self.get_tf_subset("microphones", purpose="setup", zone="dark")

        if method == 'BC':
            for i in range(len(self.frequencies)):
                ga = g_a[i]

                ga_h = ga.conj().T
                g = ga_h @ ga

                w, v = np.linalg.eigh(g)
                max_eig_val = np.argmax(w)

                q = v[:, max_eig_val]
                scale_q(q, i)

        elif method == 'ACC':

            for i in range(len(self.frequencies)):
                ga = g_a[i]
                gb = g_b[i]

                ga_h = ga.conj().T
                gb_h = gb.conj().T

                w, v = np.linalg.eig(np.linalg.solve(gb_h @ gb + beta * np.identity(gb.shape[1]), ga_h @ ga))
                max_eig_val = np.argmax(w)
                q = v[:, max_eig_val]
                scale_q(q, i)

        elif method == 'PC':

            def planarity_window(nTheta, start, end, transition):
                if abs(abs(start - end) - 360) < transition * 2:
                    print('bad window')
                    quit(1)

                win_func = np.zeros(nTheta, float)

                for i in range(start - transition, start):
                    theta = i % nTheta
                    win_func[theta] = (np.cos(np.deg2rad(180 + (180 * (transition - theta) / transition))) + 1) / 2

                for i in range(start, end + 1):
                    theta = i % nTheta
                    win_func[theta] = 1

                for i in range(end + 1, end + transition + 1):
                    theta = i % nTheta
                    win_func[theta] = (np.cos(np.deg2rad(180 + (180 * (transition + theta) / transition))) + 1) / 2

                return np.diag(win_func)

            bsv = self.steering_vectors[0]

            if bsv is None:
                print("No setup steering vector, planarity control not possible")
                return

            start_angle = self._planarity_constants[0]
            end_angle = self._planarity_constants[1]
            transition = self._planarity_constants[2]
            win = planarity_window(bsv.nTheta, start_angle, end_angle, transition)

            for i in range(len(self.frequencies)):
                ga = g_a[i]
                gb = g_b[i]

                ga_h = ga.conj().T
                gb_h = gb.conj().T

                hha = bsv.hha[i]
                hha_h = hha.conj().T

                g = np.linalg.solve(gb_h @ gb + beta * np.identity(gb.shape[1]), ga_h @ hha_h @ win @ hha @ ga)
                w, v = np.linalg.eig(g)
                max_eig_val = np.argmax(w)
                q = v[:, max_eig_val]
                scale_q(q, i)

        elif method == 'PM':

            angle = np.deg2rad((self._pm_angle-90) % 360)
            
            pos_a = setup_bright.get_object_positions()
            pos_b = setup_dark.get_object_positions()

            r_a = [np.sqrt((pos[0])**2 + ((pos[1])**2)) for pos in pos_a]
            phi_a = [np.arctan2(pos[1], pos[0]) for pos in pos_a]

            r_b = [np.sqrt((pos[0])**2 + ((pos[1])**2)) for pos in pos_b]
            phi_b = [np.arctan2(pos[1], pos[0]) for pos in pos_b]

            for i in range(len(self.k)):
                d_a = [np.exp(1j * self.k[i] * r_a[j] * np.cos(angle - phi_a[j])) for j in range(len(r_a))]
                d_b = [np.exp(1j * self.k[i] * r_b[j] * np.cos(angle - phi_b[j])) * 10**(-76 / 20) for j in range(len(r_b))]
                h = np.concatenate((g_a[i], g_b[i]), 0)
                h_h = h.conjugate().T
                d = np.array(d_a + d_b)
                q = np.linalg.solve(h_h @ h + beta * np.identity(h.shape[1]), h_h @ d)
                scale_q(q, i)

        else:
            print("Please input a valid method")
            return

        # Once the raw filter weights have been calculated they are scaled to meet the target SPL

    def calculate_metrics(self, contrast=False, effort=False, planarity=False, reproduction_error=False):
        """
        Calculate the metrics for the most recently calculated filter weights. These metrics can be chosen using the
        optional arguments. Contrast, effort, planarity and reproduction_error. Be advised ther reproduction error is only
        appropriate with pressure matching method.

        :param contrast: If true calculate the acoustic contrast between the dark and bright zone
        :type contrast: bool
        :param effort: If true calculate the difference in effort between q_ref and the calculated filter weights
        :type effort: bool
        :param planarity: If true calculate the planarity of the soundfield in the bright zone
        :type planarity: bool
        :param reproduction_error: If true calculate the reproduction error in the bright zone
        :type reproduction_error: bool
        :return: The calculated metrics in the Metrics container class

        """

        metrics = np.zeros((len(self.frequencies), 4), float)

        tfs = [0]*len(self.frequencies)
        eval_bright, tfs_bright = self.get_tf_subset("microphones", purpose="evaluation", zone="bright")
        eval_dark, tfs_dark = self.get_tf_subset("microphones", purpose="evaluation", zone="dark")
        eval_mics = eval_bright + eval_dark
        for i in range(len(self.frequencies)):
            tfs[i] = np.concatenate((tfs_bright[i], tfs_dark[i]), 0)
        self.calculate_sound_pressures(tfs, eval_mics)

        if contrast is True:
            for i in range(len(self.frequencies)):
                bright_pressure = np.array(eval_bright.get_pressures(frequency_index=i))
                dark_pressure = np.array(eval_dark.get_pressures(frequency_index=i))

                contrast_val = 10 * np.log10((len(eval_bright) * (bright_pressure.conj().T @ bright_pressure)) / \
                                             (len(eval_dark) * (dark_pressure.conj().T @ dark_pressure)))

                metrics[i][0] = np.real(contrast_val)

        if effort is True:
            for i in range(len(self.frequencies)):
                q = np.array(self.ls_array.get_q(frequency_index=i))
                effort_val = 10 * np.log10((q.conj().T @ q)/(abs(self._q_ref)**2))
                metrics[i][1] = np.real(effort_val)

        if planarity is True:
            esv = self.steering_vectors[1]
            if esv is None:
                print("No evaluation steering vector, planarity metric not available")
                return

            for i in range(len(self.frequencies)):
                p = np.array(eval_bright.get_pressures(frequency_index=i))
                dA = esv.hha[i] @ p
                eA = abs(dA ** 2)
                eA = eA / np.max(eA)
                phiInd = np.argmax(eA)
                phiHat = esv.hTheta[phiInd]
                etaA = eA * np.cos(esv.hTheta - phiHat)
                metrics[i][2] = 100 * sum(etaA) / sum(eA)

        if reproduction_error is True:
            angle = np.deg2rad((self._pm_angle-90) % 360)

            pos_a = eval_bright.get_object_positions()
            pos_b = eval_dark.get_object_positions()

            r_a = [np.sqrt((pos[0])**2 + ((pos[1])**2)) for pos in pos_a]
            phi_a = [np.arctan2(pos[1], pos[0]) for pos in pos_a]

            r_b = [np.sqrt((pos[0])**2 + ((pos[1])**2)) for pos in pos_b]
            phi_b = [np.arctan2(pos[1], pos[0]) for pos in pos_b]

            for i in range(len(self.frequencies)):
                if self._current_method is not 'PM':
                    metrics[i][3] = None
                else:

                    d_a = [np.exp(1j * self.k[i] * r_a[j] * np.cos(angle - phi_a[j])) * self._target_pa for j in
                           range(len(r_a))]
                    d_b = [np.exp(1j * self.k[i] * r_b[j] * np.cos(angle - phi_b[j])) * 10 ** (-76 / 20) for j in
                           range(len(r_b))]
                    d = np.array(d_a + d_b)

                    bright_pressure = np.array(eval_bright.get_pressures(frequency_index=i))
                    dark_pressure = np.array(eval_dark.get_pressures(frequency_index=i))
                    p = np.concatenate((bright_pressure, dark_pressure), 0)

                    error = np.subtract(p, d)
                    repro_error = error.conjugate().T @ error / (d.conjugate().T @ d)
                    metrics[i][3] = 10 * np.log10(abs(repro_error))

        return Metrics(self.frequencies, metrics)

    def add_steering_vectors(self, steering_vectors):
        """
        Add planarity steering vectors to the simulation for use in calculating Planarity Control filter weights and
        planarity evaluation metrics.

        :param steering_vectors: The input steering vector(s)
        :type steering_vectors: list

        """

        if type(steering_vectors) is SteeringVector:
            steering_vectors = [steering_vectors]

        for str_vec in steering_vectors:
            if str_vec.purpose is "setup":
                self.steering_vectors[0] = str_vec
            elif str_vec.purpose is "evaluation":
                self.steering_vectors[1] = str_vec

    def calculate_sound_pressures(self, tf_array, mic_array):
        """
        Calculate the sound pressure at each microphone across each frequency for the most recent filter weights. Sets
        the pressures of the microphones in mic_array

        :param tf_array: Transfer functions of the MicrophoneArray and LoudspeakerArray
        :type tf_array: numpy.ndarray
        :param mic_array: The MicrophoneArray used
        :type mic_array: MicrophoneArray

        """

        for i in range(len(self.frequencies)):
            q_matrix = np.array(self.ls_array.get_q(frequency_index=i))
            pressures = (tf_array[i] @ q_matrix[:, None]).flatten()
            mic_array.set_pressures(pressures, frequency_index=i)

    def get_avg_pa(self, frequency_index, purpose="either", zone="either"):
        """
        Get the average pressure in the zone across the microphones of the given purpose. The index of the frequency in
        the frequency vector must be given.

        :param frequency_index: The index of the frequency in the frequency vector
        :type frequency_index: int
        :param zone: The zone in which the pressure should be averaged - "bright", "dark" or "either"
        :type zone: str
        :param purpose: The purpose of microphones for which the pressure should be averaged -  "setup", "evaluation" or
         "either"
        :type purpose: str
        :return: The average pressure in pascals
        :rtype: float

        """

        q = np.array(self.ls_array.get_q(frequency_index=frequency_index))
        mics, subset = self.get_tf_subset("microphones", purpose=purpose, zone=zone)
        ga = subset[frequency_index]

        q_h = q.conj().T
        ga_h = ga.conj().T

        return np.sqrt(((q_h @ ga_h) @ (ga @ q)) / len(mics))


class SteeringVector:
    """
    Steering vectors are used in planarity control and evaluation.

    Attributes
    ----------
    hTheta : numpy.ndarray
        The first component of the steering vector
    hha : numpy.ndarray
        The second component of the steering vector
    nTheta : int
        The number of angles
    purpose : str
        The purpose of the steering vector. "setup" for Planarity Control or "evaluation" for the planarity metric.

    """

    def __init__(self, sim, purpose, pass_beam=3, stop_beam=6, nTheta=360, beta=0.001, file_path=None):
        """
        Create a planarity steering vector, either from scratch or from a saved steering vector csv file.

        :param sim: The simulation this will be added to
        :type sim: Simulation
        :param purpose: The purpose of the steering vector. "setup" for Planarity Control or "evaluation" for the
         planarity metric.
        :type purpose: str
        :param pass_beam: The width of the pass beam
        :type pass_beam: int
        :param stop_beam: The width of the stop beam
        :type stop_beam: int
        :param nTheta: The number of angles
        :type nTheta: int
        :param beta: A constant used in the calculation
        :type beta: float
        :param file_path: The file path at which the saved steering vector is located for loading.
        :type file_path: str

        """

        if file_path is None:
            self.hTheta, self.hha = self._make(sim, purpose, pass_beam, stop_beam, nTheta, beta)
        else:
            self.hTheta, self.hha = self._load(file_path, len(sim.k))
        self.nTheta = len(self.hTheta)
        self.purpose = purpose

    @staticmethod
    def _load(file_path, num_frequencies):
        """
        Load a steering vector from the given file path.

        :param file_path: The file path at which the saved steering vector is located for loading.
        :type file_path: str
        :param num_frequencies: The number of frequencies expected in the steering vector
        :type num_frequencies: int
        :return: The two components of the steering vector - hTheta and hha
        :rtype: numpy.ndarray, numpy.ndarray

        """

        hTheta = []
        hha = []
        nTheta = 0
        parameter = 0
        i = 0
        with open(file_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in csv_reader:
                if parameter is 0:
                    if row[0] not in ['hha', 'hTheta']:
                        hTheta.append(float(row[0]))
                        nTheta += 1
                    elif row[0] != 'hTheta':
                        parameter = 1
                elif parameter is 1:
                    if i == 0:
                        hha = np.zeros((num_frequencies, nTheta, len(row)), complex)
                    values = [complex(row[i]) for i in range(0, len(row))]
                    hha[i // nTheta][i % nTheta] = values
                    i += 1

        if i // nTheta != num_frequencies - 1:
            print("mismatch between number of frequencies in simulation and steering vector")

        return np.array(hTheta), hha

    @staticmethod
    def _make(sim, purpose, pass_beam, stop_beam, nTheta, beta):
        """
        Create a planarity steering vector

        :param sim: The simulation this will be added to
        :type sim: Simulation
        :param purpose: The purpose of the steering vector. "setup" for Planarity Control or "evaluation" for the
         planarity metric.
        :type purpose: str
        :param pass_beam: The width of the pass beam
        :type pass_beam: int
        :param stop_beam: The width of the stop beam
        :type stop_beam: int
        :param nTheta: The number of angles
        :type nTheta: int
        :param beta: A constant used in the calculation
        :type beta: float
        :return:

        """
        mics = sim.mic_array.get_subset(zone="bright", purpose=purpose)
        positions = mics.get_object_positions()
        nA = len(positions)
        k_vec = sim.k
        nK = len(k_vec)
        hha = np.zeros((nK, nTheta, nA), complex)

        hTheta = np.zeros(nTheta, float)
        eH = np.zeros((nTheta, 2), float)
        for theta in range(nTheta):
            hTheta[theta] = theta * 2 * np.pi / nTheta
            eH[theta] = [np.sin(hTheta[theta]), np.cos(hTheta[theta])]

        for i in range(len(k_vec)):
            matrix = positions @ eH.conjugate().T
            ga = np.exp(1j * k_vec[i] * matrix) / nA
            ga = np.array(ga).T

            for theta in range(nTheta):
                pass_ind = [np.mod(angle, nTheta) for angle in range(theta - pass_beam, theta + pass_beam + 1)]
                not_stop_ind = [np.mod(angle, nTheta) for angle in range(theta - stop_beam, theta + stop_beam + 1)]

                stop_ind = np.arange(0, nTheta)
                stop_ind = np.delete(stop_ind, not_stop_ind)

                gb = ga[pass_ind]
                gd = ga[stop_ind]
                gb_h = gb.conjugate().T
                gd_h = gd.conjugate().T
                g = np.linalg.solve(gd_h @ gd + beta * np.identity(nA), gb_h @ gb)
                w, v = np.linalg.eig(g)
                max_eig_val = np.argmax(w)
                hha[i][theta] = np.array(v[:, max_eig_val])
        return hTheta, hha

    def save(self, file_path):
        """
        Save a steering vector to the given file path

        :param file_path: The file path at which the steering vector will be saved
        :type file_path: str

        """

        table = [["hTheta"]]
        for angle in self.hTheta:
            table.append([str(angle)])

        table.append(["hha"])
        for f in range(self.hha.shape[0]):
            for i in range(len(self.hTheta)):
                hha_val = [str(val) for val in self.hha[f][i]]
                table.append(hha_val)

        with open(file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in table:
                csv_writer.writerow(row)


class Metrics:
    """
    Contains metrics for acoustic contrast, effort, planarity and reproduction error. This class can be used to print
    these metrics as well as being able to save them to a CSV and create a plot over frequency.

    Attributes
    ----------
    _metrics : numpy.ndarray
        An array of the metrics across frequency
    frequencies : numpy.ndarray
        The frequency vector over which the metrics have been calculated

    """

    def __init__(self, frequencies, metrics):
        """
        Create a Metrics object

        :param frequencies: The frequency vector over which the metrics have been calculated
        :type frequencies: numpy.ndarray
        :param metrics: An array of the metrics across frequency
        :type metrics: numpy.ndarray

        """

        self._metrics = metrics
        self.frequencies = frequencies

    def _output(self, contrast=False, effort=False, planarity=False, reproduction_error=False):
        """
        Creates an output table for printing or saving as a CSV for the metrics chosen.

        :param contrast: If true output the acoustic contrast between the dark and bright zone
        :type contrast: bool
        :param effort: If true output the difference in effort between q_ref and the calculated filter weights
        :type effort: bool
        :param planarity: If true output the planarity of the soundfield in the bright zone
        :type planarity: bool
        :param reproduction_error: If true output the reproduction error in the bright zone
        :type reproduction_error: bool
        :return: A table of the metrics to be output
        :rtype: list

        """

        print_list = []
        if contrast:
            print_list.append(0)
        if effort:
            print_list.append(1)
        if planarity:
            print_list.append(2)
        if reproduction_error:
            print_list.append(3)

        header_list = ['Contrast', 'Effort', 'Planarity', 'Repro Error']
        headers = [str(header_list[vals]) for vals in print_list]
        table = [['Frequency'] + headers]
        for i in range(len(self.frequencies)):
            metrics = ["{:.4f}".format(self._metrics[i][vals]) for vals in print_list]
            table.append([str(self.frequencies[i])] + metrics)

        return table

    def print(self, contrast=False, effort=False, planarity=False, reproduction_error=False):
        """
        Prints the metrics chosen to the terminal

        :param contrast: If true print the acoustic contrast between the dark and bright zone
        :type contrast: bool
        :param effort: If true print the difference in effort between q_ref and the calculated filter weights
        :type effort: bool
        :param planarity: If true print the planarity of the soundfield in the bright zone
        :type planarity: bool
        :param reproduction_error: If true print the reproduction error in the bright zone
        :type reproduction_error: bool

        """

        table = self._output(contrast=contrast, effort=effort, planarity=planarity, reproduction_error=reproduction_error)
        col_width = max(len(word) for row in table for word in row) + 2  # padding
        for row in table:
            output_format = ''.join(word.rjust(col_width) for word in row)
            print(output_format % row)

    def output_csv(self, file_name, overwrite=False, contrast=False, effort=False, planarity=False, reproduction_error=False):
        """
        Save the chosen metrics to a csv file

        :param file_name: The file path of the csv file
        :type file_name: str
        :param overwrite: Should the file be overwritten or appended to if it already exists.
        :type overwrite: bool
        :param contrast: If true save the acoustic contrast between the dark and bright zone to the csv
        :type contrast: bool
        :param effort: If true save the difference in effort between q_ref and the calculated filter weights to the csv
        :type effort: bool
        :param planarity: If true save the planarity of the soundfield in the bright zone to the csv
        :type planarity: bool
        :param reproduction_error: If true save the reproduction error in the bright zone to the csv
        :type reproduction_error: bool

        """

        table = self._output(contrast=contrast, effort=effort, planarity=planarity, reproduction_error=reproduction_error)

        if overwrite:
            open_as = 'w'
        else:
            open_as = 'a'

        with open(file_name, open_as, newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in table:
                csv_writer.writerow(row)

    def plot(self, graph_name, file_name, metric):
        """
        Plot the chosen metric over frequency

        :param graph_name: The name of the graph
        :type graph_name: str
        :param file_name: The file path of the image made
        :type file_name: str
        :param metric: The metric chosen - 'contrast', 'effort', 'planarity' or 'reproduction error'
        :type metric: str

        """

        if metric is 'contrast':
            ind = 0
        elif metric is 'effort':
            ind = 1
        elif metric is 'planarity':
            ind = 2
        elif metric is 'reproduction error':
            ind = 3
        else:
            print('please choose an available metric')
            return

        plot_metric = np.swapaxes(self._metrics, 0, 1)[ind]
        fig = plt.figure(figsize=(6, 6), dpi=1000)
        axes = fig.add_subplot(111)
        axes.semilogx(self.frequencies, plot_metric)

        if plot_metric.min() < 0:
            axes.set_ylim(bottom=plot_metric.min(), top=0)
        else:
            axes.set_ylim(bottom=0)

        if len(self.frequencies) > 1:
            axes.set_xlim(left=self.frequencies[0], right=self.frequencies[-1])

        axes.set_xlabel('Frequency (Hz)')
        axes.set_ylabel('Decibels')
        axes.set_title(graph_name)
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        fig.savefig(file_name)

# ********* General functions *********


def convert_to_db(pressures):
    """
    Convert a pressure or list of pressures to decibels

    :param pressures: The input pressures
    :type pressures: float
    :return: The resultant pressures in dB
    :rtype: float

    """

    return 20*np.log10(pressures/0.00002)



