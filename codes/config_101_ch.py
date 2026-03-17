import os
from convolutional_synth_3d import ConvolutionalSynth

workdir='../'

# Read already existing catalogue
event_dir=os.path.join(workdir,'CAT')
filename_events='catalogue_flegrei_MT_final.txt'                                ### CHANGE ###
events_file=os.path.join(event_dir,filename_events)

#### 2 - FIBER GEOMETRY (load)
fiber_geometry_dir=os.path.join(workdir,'FIBER_GEOMETRY')
fiber_geometry_file=os.path.join(fiber_geometry_dir, 'flegrei_das_geom_101ch.txt')      ### CHANGE ###

#### 3 - NLL traveltime/angle matrices (load)
db_path = workdir + 'NLL/FLEGREI_fiber_101/nll_grid'                                                    ### CHANGE ###
hdr_filename = 'header.hdr'
precision='single'
label = 'time'
NLL_matrices_files = {
    'db_path': db_path, 'hdr_filename': hdr_filename,
    'precision': precision, 'label': label}

#### 4 - TIME AXIS (generate)
dt= 0.01  # == 100 Hz                           # ARBITRARY (?) 
time_window= 10 #s after origin time            # CHANGE

#### 5 - RICKER WAVELET (generate)
frequency_w=3.                                  # CHANGE
time_window_w=1. # s                           # CHANGE
dt_w=None                                      # if None, use dt
derivative_w=True                             # if True, use derivative of Ricker

time_inputs={
    'dt':dt, 'time_window':time_window,
    'frequency_w':frequency_w, 'time_window_w':time_window_w ,
    'dt_w':dt_w, 'derivative_w':derivative_w}

#%% SYNTHETIC GENERATION:

# ConvolutionalSynth class
synth_class=ConvolutionalSynth(events_path = events_file, # 1 - CATALOGUE
                            fiber_geometry_path = fiber_geometry_file, # 2 - FIBER GEOMETRY
                            NLL_matrices_path = NLL_matrices_files, # 3 - NLL MATRICES
                            time_parameters = time_inputs) # 4,5 - TIME AXIS + RICKER WAVELET

#----------------------------------------------------------------------
# ih: incidence angle !!!MISSING!!!

# phir: reciver-azimuth !!!MISSING!!!

#----------------------------------------------------------------------
# synthetic seismogram of ALL events
#synth_class.generate_synthetics(noise_type='gaussian', file_prefix='', plot_fig=False, save_fig=False, save_mseed=False, save_npy=True)

# synthetic seismogram of one event
ev_number=52         # CHANGE
seis = synth_class.convolution(synth_class.events[ev_number],noise_type='none')

synth_class.plot_seismogram(seis,synth_class.events[ev_number], file_prefix='', plot_fig=True, save_fig=False)

synth_class.save_seismogram(seismogram = seis,
                            event = synth_class.events[ev_number],
                            file_prefix='synth_101ch_0_',
                            save_mseed=True,save_npy=False)
