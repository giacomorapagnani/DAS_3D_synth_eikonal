import numpy as np
from ctypes import Union, c_float, c_ushort
import matplotlib.pyplot as plt
import sys
import os

# ---- Union equivalent to NonLinLoc TakeOffAngles ----
class TakeOffAngles(Union):
    _fields_ = [
        ("fval", c_float),
        ("ival", c_ushort * 2),
    ]

class Angles_NLL:
    def __init__(self, nll_path, hdr_name, label):
        self.nll_path = nll_path
        self.label=label
        
        self._read_nll_header(nll_path, hdr_name)



    # ---- Read header file ----
    def read_hdr(self,hdr_file):
        with open(hdr_file, "r") as f:
            first_line = f.readline().split()

        nx = int(first_line[0])
        ny = int(first_line[1])
        nz = int(first_line[2])
        grid_type = first_line[9]

        return nx, ny, nz, grid_type


    # ---- Read ANGLE buffer ----
    def read_angle_buf(self,buf_file, nx, ny, nz):

        nitems = nx * ny * nz

        # read raw floats
        buf = np.fromfile(buf_file, dtype=np.float32, count=nitems)

        if len(buf) != nitems:
            raise ValueError("Incorrect number of values in buffer, not matching header grid.")

        # reinterpret floats using union
        angles = (TakeOffAngles * nitems)()

        for i, val in enumerate(buf):
            angles[i].fval = val

        azimuth = np.array(
            [a.ival[1] / 10.0 for a in angles]
        ).reshape(nx, ny, nz)

        dip = np.array(
            [(a.ival[0] // 16) / 10.0 for a in angles]
        ).reshape(nx, ny, nz)

        quality = np.array(
            [a.ival[0] % 16 for a in angles]
        ).reshape(nx, ny, nz)

        # apply NonLinLoc convention
        azimuth[quality == 0] = np.nan
        dip[quality == 0] = np.nan

        return azimuth, dip, quality

    def read_all_angles(self,phase,quality=False):
        angles={}
        for sta in self.stat_list:
            try:
               fn_hdr = os.path.join(self.nll_path, '%(label)s.%(phase)s.%(station)s.angle.hdr' %{"label":self.label,"phase":phase, "station":sta} )
               fn_buf = os.path.join(self.nll_path, '%(label)s.%(phase)s.%(station)s.angle.buf' %{"label":self.label,"phase":phase, "station":sta} )
            except:
               print(f'Error while reading file for station {sta}.\nCheck path for angles files: {fn_buf}')
               sys.exit()

            nx, ny, nz, grid_type = self.read_hdr(fn_hdr)
            if grid_type != 'ANGLE':
                print(f'ERROR: not ANGLE file {fn_hdr}')
                sys.exit()
            
            if quality:
                angles[sta] = {'az': None, 'dip': None, 'qual': None}
                angles[sta]['az'], angles[sta]['dip'], angles[sta]['qual']= self.read_angle_buf(fn_buf, nx, ny, nz)
            else:
                angles[sta] = {'az': None, 'dip': None}
                angles[sta]['az'], angles[sta]['dip'], _ = self.read_angle_buf(fn_buf, nx, ny, nz)
        print(f'Successfully loaded ANGLE data for {len(self.stat_list)} channels (phase {phase})')
        return angles

    def _read_nll_header(self, nll_path, hdr_name):
        if not os.path.isdir(nll_path):
            print('Error: NLL database path do not exist')
            sys.exit()

        f = open(os.path.join(nll_path, hdr_name))
        lines = f.readlines()
        f.close()
        # read only stations list from NLL header
        stat_list=[]
        for line in lines[4:]:
            toks=line.split()
            stat_list.append(toks[0])
        self.stat_list=stat_list

        return None


# ---- Example usage ----
if __name__ == "__main__":

    nll_database='../NLL/FLEGREI_stations/nll_grid'
    header_name='header.hdr'
    label='time'

    angles_class = Angles_NLL(nll_database, header_name, label)

    phase='P'
    angles = angles_class.read_all_angles(phase=phase,quality=False)

    # angles for first starion
    st='CAAM'
    
    ###
    # Tests
    ###
    az = angles[st]['az']
    dip = angles[st]['dip']

    print("Azimuth shape:", az.shape)
    print("Dip shape:", dip.shape)

    print(np.nanmin(az))
    print(np.nanmax(az))

    plt.imshow(az[:,:,30])
    plt.colorbar()
    plt.show()