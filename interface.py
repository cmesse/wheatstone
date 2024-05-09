import numpy as np
import scipy as sp
import scipy.sparse


def create_node_adjacency( ElementConnectivity: np.ndarray, Dirichlet: np.ndarray ) :
    # total number of resistor elements
    numElements = len(ElementConnectivity)

    # total number of nodes
    numNodes = np.max(ElementConnectivity) + 1
    numDofs    = numNodes - len( Dirichlet )

    isFree = np.ones( numNodes, dtype=int )
    for d in Dirichlet :
        isFree[d] = 0

    # create the new index
    dofIndex = np.zeros( numNodes, dtype=int )

    i = 0
    j = 0
    for k in range( numNodes ) :
        if isFree[k] :
            dofIndex[k] = i
            i+=1
        else :
            dofIndex[k] = j
            j+=1

    # first, we count the number of elements per node
    numElementsPerDof = np.zeros(numDofs, dtype=int)

    for e in range(numElements):
        for i in range(2):
            j = ElementConnectivity[e][i]
            if isFree[j]:
                k = dofIndex[j]
                numElementsPerDof[k] += 1


    # now we allocate the memory for elements per node
    ElementsPerDof = [None] * numDofs
    for k in range(numDofs):
        ElementsPerDof[k] = np.zeros(numElementsPerDof[k], dtype=int)


    # reset the counters
    numElementsPerDof = np.zeros(numDofs, dtype=int)
    for e in range(numElements):
        for i in range(2):
            j = ElementConnectivity[e][i]
            if isFree[j]:
                k = dofIndex[j]
                ElementsPerDof[k][numElementsPerDof[k]] = e
                numElementsPerDof[k] += 1

    # count dofs per dof
    dofsPerDof = [ None ] * numDofs

    nodeFlags = np.zeros( numNodes, dtype=bool )

    for d in range( numDofs ) :
        # count dofs
        c = 0
        for e in ElementsPerDof[d]:
            for j in ElementConnectivity[e] :
                if isFree[j] and not nodeFlags[j] :
                    c+=1
                    nodeFlags[j] = True

        # allocate memory
        dofsPerDof[d] = np.zeros( c, dtype=int )
        c = 0

        # populate dofs
        for e in ElementsPerDof[d]:
            for j in ElementConnectivity[e] :
                if isFree[j] and nodeFlags[j] :
                    dofsPerDof[d][c] = dofIndex[j]
                    c+=1
                    nodeFlags[j] = False

        dofsPerDof[d] = np.sort( dofsPerDof[d] )

    return dofsPerDof, dofIndex, isFree


def create_matrix( nodesPerNode: np.ndarray ):
    numNodes = len( nodesPerNode )

    # create the pointers
    pointers = np.zeros( numNodes + 1, dtype=int )
    c = 0
    k = 0
    for n in nodesPerNode :
        pointers[k] = c
        k+=1
        c+=len( n )
    pointers[numNodes] = c

    # number of nonzeros
    nnz = c
    c = 0
    k = 0
    indices = np.zeros( nnz, dtype=int )

    for n in nodesPerNode:
        for i in n :
            indices[c] = i
            c+=1
    values = np.zeros( nnz, dtype=float )

    return scipy.sparse.csc_matrix( ( values, indices, pointers) , shape=( numNodes, numNodes ) )


def addval(K: scipy.sparse.csc_matrix, i: int, j: int, value: float ):
    """
    Add a value to a specific position in a scipy.sparse.csc_matrix.

    Parameters:
        K (scipy.sparse.csc_matrix): The matrix to modify.
        i (int): Row index of the value to modify.
        j (int): Column index of the value to modify.
        value (numeric): Value to add to K[i, j].
    """

    # Find the position in the data array that corresponds to K[i, j]
    col_start = K.indptr[j]
    col_end = K.indptr[j + 1]

    # Find the row index within the current column
    data_idx = col_start + np.where(K.indices[col_start:col_end] == i)[0][0]
    K.data[data_idx] += value


def compute_system( Elements: np.ndarray,
                    Values: np.ndarray,
                    Dirichlet: np.ndarray,
                    Adj: np.ndarray,
                    dofIndex: np.ndarray,
                    free: np.ndarray ) :

    # the system matrix
    K = create_matrix( Adj )

    # the right hand side
    f = np.zeros( len( K.indptr ) - 1, dtype=float )

    # default matrix
    K_el = np.zeros( ( 2,2 ), dtype=float )

    # connection matrix
    C_el = np.zeros( ( 2,2 ), dtype=float )
    C_el[0][0] =  1
    C_el[0][1] = -1
    C_el[1][0] = -1
    C_el[1][1] =  1

    c = 0
    for e in Elements :
        K_el = C_el * ( 1 / Values[c] )
        c+=1
        for i in range( 2 ) :
            k = e[i]
            if free[k]:
                m = dofIndex[k]
                for j in range(2):
                    l = e[j]
                    n = dofIndex[l]
                    if free[l]:
                        addval( K, m, n, K_el[i][j] )
                    else :
                        f[m] -= Dirichlet[n]*K_el[i][j]


                        # K[m][n] += K_el[i][j]

    return K, f


def recombine( dofIndex: np.ndarray,
               free: np.ndarray,
               dirichlet: np.ndarray,
               freevalues: np.ndarray ):

    n = len( dofIndex )
    values = np.zeros( n, dtype=float )

    for k in range( n ) :
        if free[k] :
            values[k] = freevalues[dofIndex[k]]
        else:
            values[k] = dirichlet[dofIndex[k]]

    return values
