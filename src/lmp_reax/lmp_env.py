#!/usr/bin/env python3

"""
this file stores static variables for MD post
"""

# the "type" indicates atom type in lmp_data_file,
# where type is a local index in lmp, need convect to real name.
type2element_str_dict = {0: "Gst", 1: "C", 2: "H", 3: "O", 4: "N", 5: "N"}
type2element_id_dict = {0: 0, 1: 6, 2: 1, 3: 8, 4: 7}
BLOCK_SIZE=[100]
# if the atom number greater than this val, smile parser will ignore it
MAX_SMILE_MOLE_NUM = 30
# average the result each 0.01ns, 1e4 ps is 0.01ns
OUTPUT_TIME_STEP_NUM = 1e2
# specie enlarge for small val
SPECIE_ENLARGE = 1e3
# path threshold
PATH_THRESHOLD = 2
PATH_MAX_NUM = 255
PATH_MIN_NUM = 3

# rxn path species
# RXN_START_SPECIE = ['C10H8__0800r6-2'] # for PAH case
RXN_START_SPECIE = ['CH4__0400']

FLUX_SPECIE = ['H', "OH", "HO", 'H2O', 'H2']
SKIP_FRAME = [1]
# reaction rate sample section, unit ns
METAL = [False]
