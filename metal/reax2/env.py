from dataclasses import dataclass


@dataclass
class EnvFrame:
    # if the atom number greater than this val, smile parser will ignore it
    # the "type" indicates atom type in lmp_data_file,
    # where type is a local index in lmp, need convect to real name.

    # carbon system parameter
    # type2element_str_dict = {0: "Gst", 1: "C", 2: "H", 3: "O", 4: "N"}
    # type2element_id_dict = {0: 0, 1: 6, 2: 1, 3: 8, 4: 7}

    type2element_str_dict = {1: "Al", 2: "H", 3: "O", 4: "N"}
    element_stro2typedict = {"Al": 1, "H": 2, "O": 3, "N": 4}
    type2element_id_dict = {1: 13, 2: 1, 3: 8, 4: 7}
    MAX_SMILE_MOLE_NUM = 30
    METAL = True
    METAL_ID = 1
    OXYGEN_ID = 3
    METAL_NAME = 'Al'
    # use for cords system to ref whether bonded or not
    # ref DOI: 10.1021/ic0617487 Al-AL:2.89
    distance_sq_ref = {(1, 1): 2.89, (1, 2): 1.79, (1, 3): 1.97,
                       (2, 2): 0.737, (2, 3): 0.976,
                       (3, 3): 1.188849}
    # enlarge with 20%, use int compare x100
    distance_sq_ref = {x: (y * 1.2) ** 2 for x, y in distance_sq_ref.items()}
    changes = {}
    for x, y in distance_sq_ref.items():
        rev_x = x[1], x[0]
        if distance_sq_ref.get(rev_x, None) is None:
            changes[rev_x] = y
    distance_sq_ref.update(changes)


@dataclass
class EnvBlock:
    CACHE = True
    skip_frame = 1
    worker = 8
    sub_block_size = 100  # should be multiple or worker
    key_reaction = 'HO->#metal#(HO)'
    key_reaction_re = "->".join(key_reaction.split("->")[::-1])


@dataclass
class EnvCase:
    DataSourceNC = True  # load data from NetCDF
    # DataSourceNC = False  # load data from NetCDF
    SampleVal = 1
    MinSpecieVal = 0.1
    RXN_START_SPECIE = "H2O"
    SPECIE_ENLARGE = 1e3
    OUTPUT_TIME_STEP_NUM = 1e4  # unit fs, average the result each 0.01ns, 1e4 ps is 0.01ns
    RATE_TIME_SECTION = 1  # reaction rate sample section, unit ns
    FLUX_SPECIE = ('H', "OH", "HO", 'H2O', 'H2')
    PATH_THRESHOLD = [2 ** x for x in range(11)]
    PATH_MAX_NUM = 255
    PATH_MIN_NUM = 3
