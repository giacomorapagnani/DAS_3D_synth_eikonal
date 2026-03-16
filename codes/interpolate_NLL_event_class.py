import os
import numpy as np
from traveltimes_NLL_class import Traveltimes_NLL
from angles_NLL_class import Angles_NLL
import matplotlib.pyplot as plt
from latlon2cart_class import Coordinates
from scipy.interpolate import RegularGridInterpolator
import sys

class InterpolateNLLMatrix:
    def __init__(self, NLL_path, fiber_geometry_path):
        # load fiber geometry
        self.fiber_geometry = self._load_fiber_geometry(fiber_geometry_path)
        # laod header parameters
        self._load_header(NLL_path['db_path'],NLL_path['hdr_filename'])
        # load latlon2cart class
        self.coord=Coordinates(self.nll_par['co_lat'], self.nll_par['co_lon'],  self.nll_par['co_ele'])

    ########################################################
    ####################### LOAD ###########################
    ########################################################
    def _load_header(self,nll_path,hdr_name):

        f = open(os.path.join(nll_path, hdr_name))
        lines = f.readlines()
        f.close()
        nx, ny, nz = [ int(x)   for x in lines[0].split()] # number of grid points in x,y,z direction
        x0, y0, z0 = [ float(x) for x in lines[1].split()] # origin of x,y,z coordinate (km)
        dx, dy, dz = [ float(x) for x in lines[2].split()] # grid spacing in x,y,z (km)
        lat0, lon0, el0 = [ float(x) for x in lines[3].split()] # grid origin latitude,longitude,elevation
        self.nll_par = {
            'nx': nx,'ny': ny, 'nz': nz, 
            'dx': dx, 'dy': dy, 'dz': dz,
            'co_x': x0, 'co_y': y0, 'co_z': z0, 
            'co_lat': lat0, 'co_lon': lon0, 'co_ele': el0}
        return

    def _load_fiber_geometry(self, filepath):
        # FIBER GEOMETRY (list) ->  channel_name, lat, lon, elev
        fiber_geometry = []
        with open(filepath, "r") as f:
            next(f)  # skip header
            for line in f:
                ntw_name,st_name, lat, lon, elev = line.split()
                fiber_geometry.append([ntw_name, st_name, float(lat), float(lon), float(elev)])
        return fiber_geometry

    def _load_events(self, filepath):
        # EVENT (list) ->  event_name, tor, lat, lon, depth, mag, strike, dip, rake
        events = []
        with open(filepath, "r") as f:
            next(f)  # skip header
            for line in f:
                event_name, tor, lat, lon, depth, mag, strike, dip, rake = line.split()
                events.append([event_name, str(tor), float(lat), float(lon), float(depth), 
                               float(mag), float(strike), float(dip), float(rake)])
        return events
    ########################################################

    ########################################################
    ################ TRAVEL TIMES or ANGLE #################
    ########################################################
    def _gen_axis(self):
        # Build coordinate axes
        self.x_ax = self.nll_par['co_x'] + np.arange(self.nll_par['nx']) * self.nll_par['dx']
        self.y_ax = self.nll_par['co_y'] + np.arange(self.nll_par['ny']) * self.nll_par['dy']
        self.z_ax = self.nll_par['co_z'] + np.arange(self.nll_par['nz']) * self.nll_par['dz']
        return

    def _compute_event_coord(self,ev_lat,ev_lon,ev_depth):
        source_x,source_y,source_z = self.coord.geo2cart(ev_lat, ev_lon, ev_depth) 
        source_x,source_y,source_z = source_x * 1e-3, source_y * 1e-3, source_z * 1e-3  # convert from m to km
        return source_x, source_y, source_z

    def get_travel_time_or_angle(self, event, nll_matrices, nll_matrix_type, interpolation='linear'):
        # travel time for single event
        # EVENT (list) -> event_name, tor, lat, lon, depth, mag, strike, dip, rake
        ev_lat = event[2]  #event latitude
        ev_lon = event[3]  #event longitude
        ev_depth = event[4]  #event depth in km (positive DOWN) 
        self.ev_location = self._compute_event_coord(ev_lat,ev_lon,ev_depth)
        self._gen_axis()

        if nll_matrix_type=='TravelTime':
            travel_times=[]
            for i in range (len(self.fiber_geometry)):
                ch_name=self.fiber_geometry[i][1]
                travel_times.append( self._compute_travel_time(ch_name, nll_matrices, interpolation) )
            travel_times=np.array(travel_times, dtype=np.float32)
            return travel_times
        
        elif nll_matrix_type=='Angle':
            angles=[]
            for i in range (len(self.fiber_geometry)):
                ch_name=self.fiber_geometry[i][1]
                angles.append( self._compute_angle(ch_name, nll_matrices, interpolation) )
            azimuth, dip = np.array(angles).T
            return azimuth,dip
        else:
            print(f'Error: NLL matrix type not selected correctly.\nChoose between "TravelTime" or "Angle"')
            sys.exit()

    def _compute_travel_time(self, ch_name, nll_matrices, interpolation):
        # POSSIBLE OPTIMIZATION: avoid reshape
        cube= np.reshape(nll_matrices[ch_name],(self.nll_par['nx'], self.nll_par['ny'], self.nll_par['nz']))
        tt_or_angl=self._interpolate(cube, self.ev_location, interpolation)
        return tt_or_angl
    
    def _compute_angle(self, ch_name, nll_matrices, interpolation):
        cube_az= nll_matrices[ch_name]['az']
        cube_dip= nll_matrices[ch_name]['dip']
        az=  float( self._interpolate(cube_az,  self.ev_location, interpolation)  )
        dip= float( self._interpolate(cube_dip, self.ev_location, interpolation)  )
        return [az,dip]

    def _interpolate(self, matrix, query_points, interpolation):
        """
        Interpolates values from a 3D regular grid at arbitrary query points.
        """
        # Build interpolator (linear by default, 'nearest' also available)
        interpolator = RegularGridInterpolator(
            (self.x_ax, self.y_ax, self.z_ax),
            matrix,
            method= interpolation,
            bounds_error=True,   # set True to raise error if point is outside grid
            fill_value=np.nan     # value returned for out-of-bounds points
        )

        query_points = np.atleast_2d(query_points)  # handle single point case
        values = interpolator(query_points)          # shape (N,)

        return values
    ########################################################

if __name__ == "__main__":
    ########################################################################
    ########################### INPUTS  ####################################
    ########################################################################

    ######### FIBER GEOMETRY
    fiber_geometry_dir='../FIBER_GEOMETRY'
    fiber_geometry_file=os.path.join(fiber_geometry_dir, 'flegrei_stations_geometry.txt') ### CHANGE ###
    #########

    #########- NLL path
    nll_db_path = '../NLL/FLEGREI_stations/nll_grid'      ### CHANGE ###
    hdr_name = 'header.hdr'
    label = 'time'
    precision='single'
    NLL_matrices_path = {
        'db_path': nll_db_path, 'hdr_filename': hdr_name,
        'precision': precision, 'label': label}
    #########

    ######### EVENTS
    event_dir='../CAT'
    filename_events='catalogue_flegrei_MT_final.txt'         ### CHANGE ###
    events_file=os.path.join(event_dir,filename_events)
    #########

    ########################################################################
    ########################## EXAMPLES ####################################
    ########################################################################
    switch_get_tt=False
    if switch_get_tt:
        ######### NLL traveltimes
        tt_nll_obj = Traveltimes_NLL(nll_db_path, hdr_name)
        tt_nll_p = tt_nll_obj.load_traveltimes('P', label, precision)
        tt_nll_s = tt_nll_obj.load_traveltimes('S', label, precision)
        #########

        tt_class=InterpolateNLLMatrix(NLL_path= NLL_matrices_path,
                                    fiber_geometry_path=fiber_geometry_file)
        
        events=tt_class._load_events(events_file)
        event=events[0]  # first event

        tt_p=tt_class.get_travel_time_or_angle(event, tt_nll_p, nll_matrix_type= 'TravelTime')
        tt_s=tt_class.get_travel_time_or_angle(event, tt_nll_s, nll_matrix_type= 'TravelTime')
        ########################################################################
        ########################################################################

        ######### PLOT
        plt.figure(figsize=(10,5))
        ch_names=[item[1] for item in tt_class.fiber_geometry]
        plt.plot(ch_names, tt_p, label='P-wave Travel Time', marker='o')
        plt.plot(ch_names, tt_s, label='S-wave Travel Time', marker='o')
        plt.xlabel('Channel Name')
        plt.ylabel('Travel Time (s)')
        plt.title('Travel Times for Event: {}'.format(event[0]))
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        #########

    switch_get_angle=True
    if switch_get_angle:
        ######### NLL angles
        angles_obj = Angles_NLL(nll_db_path, hdr_name, label)
        phase='P'
        nll_angles_matrices = angles_obj.read_all_angles(phase=phase,quality=False)
        #########

        interp_obj=InterpolateNLLMatrix(NLL_path= NLL_matrices_path,
                                    fiber_geometry_path=fiber_geometry_file)
        
        events=interp_obj._load_events(events_file)
        event=events[0]  # first event

        azimuth, dip = interp_obj.get_travel_time_or_angle(event, nll_angles_matrices, nll_matrix_type= 'Angle')
        ########################################################################
        ########################################################################

        ######### PLOT
        plt.figure(figsize=(10,5))
        ch_names=[item[1] for item in interp_obj.fiber_geometry]
        plt.plot(ch_names, azimuth, label='Azimuth', marker='o')
        plt.plot(ch_names, dip, label='Dip', marker='o')
        plt.xlabel('Channel Name')
        plt.ylabel('Angle [°]')
        plt.title('Angles for Event: {}'.format(event[0]))
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        #########
