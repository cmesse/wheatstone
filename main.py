import numpy as np
import scipy.sparse.linalg

from interface import *

# This example computes the system matrix for the
# following Resistor mesh :
#
#                (-)
#                 |
#   (1)---[R0]---(0)---[R1]---(2)
#   |                         |
#  [R6]                      [R7]
#   |                         |
#  (3)---[R2]---(4)---[R3]---(5)
#   |                         |
#  [R8]                      [R9]
#   |                         |
#  (6)---[R4]---(8)---[R5]---(7)
#                |
#               (+)


# First, we create the resistor connectivity
# note, as long as we are not using diodes,
# the orientation does not matter

Rconn   = np.zeros( (10,2), dtype=int )
Rconn[0] = [ 1, 0 ]
Rconn[1] = [ 0, 2 ]
Rconn[2] = [ 3, 4 ]
Rconn[3] = [ 4, 5 ]
Rconn[4] = [ 6, 8 ]
Rconn[5] = [ 8, 7 ]
Rconn[6] = [ 1, 3 ]
Rconn[7] = [ 2, 5 ]
Rconn[8] = [ 3, 6 ]
Rconn[9] = [ 5, 7 ]

# next, we need to assign the resistor values in Ohm

Rval   = np.zeros( 10, dtype=float )
Rval[0] = 100
Rval[1] = 100
Rval[2] = 100
Rval[3] = 100
Rval[4] = 100
Rval[5] = 100
Rval[6] = 100
Rval[7] = 100
Rval[8] = 100
Rval[9] = 100

# voltages are imposed as dirichlet conditions
# currents are imposed as neumann conditions

# we always need at least one fixed voltage for the ground level

# in this example:
#   * mode==0 applies a voltage of 1.2 kV to node 8,
#   * mode==1 applies a current of 6 A to node 8,
#
# both cases should return the same result

mode = 1

if mode == 0 :
    # dirichlet boundary conditions
    Ufix  = np.zeros( 2, dtype=int )
    Uval  = np.zeros( 2, dtype=float  )

    # neumann boundary conditions
    Ifix = []
    Ival = []

    # set ground
    Ufix[0] = 0
    Uval[0] = 0

    # set voltage on second node
    Ufix[1] = 8
    Uval[1] = 1200
elif mode == 1 :
    # dirichlet boundary conditions
    Ufix = np.zeros(1, dtype=int)
    Uval = np.zeros(1, dtype=float)

    # set ground
    Ufix[0] = 0
    Uval[0] = 0

    # neumann boundary conditions
    Ifix  = np.zeros( 1, dtype=int )
    Ival  = np.zeros( 1, dtype=float  )

    # set current
    Ifix[0] = 8
    Ival[0] = 6


###################### END OF USER INPUT

# first, we build the adjacency
Adj, idx, free = create_node_adjacency( Rconn, Ufix )

# build the matrix and the right hand side
K, f = compute_system( Rconn, Rval, Uval, Adj, idx, free )

# add current BCs
c=0
for k in Ifix:
    f[ idx[ k ] ] = Ival[c]
    c+=1

# per default, this calls the umfpack library which should be sufficiently fast
x = scipy.sparse.linalg.spsolve( K, f )

# now, we need to recombine the voltages
u = recombine( idx, free, Uval, x )

##### Output
print( "\nVoltages" )

for k in range( len(u) ):
    print( "node {:2g} : {:8.3f} V".format( k, u[k]) )

print( "\nCurrents" )

c=0
for e in Rconn :
    deltaU = u[e[1]]-u[e[0]]
    I = abs(deltaU / Rval[c])
    print("resistor {:2g} : {:8.3f} A".format(c, I))
    c += 1