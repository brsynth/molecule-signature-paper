###############################################################################
# This file provide utilities
# Authors: Jean-loup Faulon jfaulon@gmail.com
# Jan 2023
###############################################################################

from library.imports import *

###############################################################################
# read write txt file where Data is a list
###############################################################################


def read_txt(filename):
    with open(filename, "r") as fp:
        Lines = fp.readlines()
        Data, i = {}, 0
        for line in Lines:
            Data[i] = line.strip()
            i += 1
        return list(Data.values())


def write_txt(filename, Data):
    with open(filename, "w") as fp:
        for i in range(len(Data)):
            fp.write("%s\n" % Data[i])


###############################################################################
# read write csv file with panda where Data is a np array
###############################################################################


def read_csv(filename):
    # Reading datafile with pandas
    # Return HEADER and DATA
    if not filename.endswith(".csv"):
        filename += ".csv"
    dataframe = pandas.read_csv(filename, header=0)
    HEADER = dataframe.columns.tolist()
    dataset = dataframe.values
    DATA = np.asarray(dataset[:, :])
    return HEADER, DATA


def write_csv(filename, H, D):
    # H = Header, D = Data
    filename += ".csv"
    with open(filename, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        # write the header
        if H != None:
            writer.writerow(H)
        # write the data
        for i in range(D.shape[0]):
            writer.writerow(D[i])
    return


def read_tsv(filename):
    # Reading datafile with pandas
    # Return HEADER and DATA
    filename += ".tsv"
    dataframe = pandas.read_csv(filename, header=0, sep="\t")
    HEADER = dataframe.columns.tolist()
    dataset = dataframe.values
    DATA = np.asarray(dataset[:, :])
    return HEADER, DATA


def write_tsv(filename, H, D):
    # H = Header, D = Data
    filename += ".tsv"
    with open(filename, "w", encoding="UTF8") as f:
        writer = csv.writer(f, delimiter="\t")
        # write the header
        if H != None:
            writer.writerow(H)
        # write the data
        for i in range(D.shape[0]):
            writer.writerow(D[i])
    return


###############################################################################
# other utilities
###############################################################################


def ReLU(x):
    # x is a nparray
    return x * (x > 0)


def VectorToDic(V):
    # Transfert a 1D array into dictionary
    # The key is array content
    # and the value is the index
    D = {}
    for i in range(len(V)):
        D[V[i]] = i
    return D


def PrintMatrix(A):
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[1]):
            if A[i, j]:
                print(f"A ({i}, {j})")
