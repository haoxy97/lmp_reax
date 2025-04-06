variable  Tstart           equal 500
variable  Tfinal           equal 1000
variable  heat_step        equal 100000
variable  StartCycle       equal 1
variable  sampleStep       equal 10000000
variable  DumpFreq         equal 100
units real
boundary    p p p
timestep 0.2
atom_style charge
read_data data.data
velocity all create  ${Tstart} 1 dist gaussian

neighbor 2 bin
neigh_modify every 10 delay 0 check no
pair_style reaxff NULL safezone 50 mincap 500
pair_coeff	* * Al_water.ff  Al H O
fix  1base   all qeq/reaxff 1 0.0 10.0 1e-6 reaxff
thermo 1000

minimize 0.0 1.0e-6 1000 1000
reset_timestep 0 # reset minina step number
group water type 2 3 
group al type 1
# initial temperature balance
fix  11 water nvt     temp  ${Tstart}   ${Tstart}  $(200.*dt) 
fix  12 al    nvt     temp  ${Tstart}   ${Tstart}  $(200.*dt) 
run   1000 # test speed
run   ${heat_step}
unfix 11
unfix 12

# heat progress with NVT
fix  1 all temp/berendsen ${Tstart}  ${Tfinal}  $(200.*dt) 
fix  2 all nve
run   ${heat_step}
unfix 1
unfix 2

fix  1 all nvt   temp  ${Tfinal}       ${Tfinal}      $(200.*dt) 
dump 2 all netcdf ${DumpFreq}  1/1.nc  id type x y z000
run  ${sampleStep}
unfix 1
undump 2