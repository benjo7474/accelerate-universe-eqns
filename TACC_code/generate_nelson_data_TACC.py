from mpi4py import MPI
import numpy as np
from nelson_langer_network import build_nelson_network
import sys

filename = sys.argv[1]
n_params = 3

# initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


### BROADCAST STEP 

if rank == 0:
    sendbuf = np.load(filename)

    # compute the size of each subtask
    ave, res = divmod(len(sendbuf), size)
    count_rows = [ave + 1 if p < res else ave for p in range(size)]
    count_rows = np.array(count_rows, dtype=int) # number of floats in each array

    displ_rows = [sum(count_rows[:p]) for p in range(size)]
    displ_rows = np.array(displ_rows, dtype=int) # array index to start at

else:
    sendbuf = None
    count_rows = np.empty(size, dtype=int)
    displ_rows = None

# broadcast count so that every process knows how much data it receives
comm.Bcast(count_rows, root=0)

# initialize recvbuf on all processes
recvbuf = np.empty(shape=(count_rows[rank], n_params), dtype=np.float64)

# broadcast
if rank == 0:
    count = n_params * count_rows
    displ = n_params * displ_rows
else:
    count = n_params * count_rows
    displ = None
comm.Scatterv([sendbuf, count, displ, MPI.DOUBLE], recvbuf, root=0)
# parameter data per process is saved in recvbuf for processing.


### PROCESSING STEP

# this function takes in ONE row of parameters and spits out q and dqdp as np arrays, separately
def solve_for_sensitivities(p: np.ndarray, QoI: int, x0: np.ndarray, teval: np.ndarray):
    NL = build_nelson_network(p, compute_sensitivities=True)
    _, eval = NL.solve_reaction([0, teval[-1]], x0, t_eval=teval)
    sens_ind = 14 * np.arange(1,4) + QoI
    q = eval[QoI]
    dqdp = eval[sens_ind]
    return q, dqdp

# parameters for solves
x0 = np.array([
    0.5,
    9.059e-9,
    2e-4,
    0.1,
    7.866e-7,
    0.0,
    0.0,
    0.0004,
    0.0,
    0.0,
    0.0,
    0.0002,
    2.0e-7,
    2.0e-7
])
tf_years_eval = np.array([100, 1000, 10000, 40000])
tf_eval = tf_years_eval * 365 * 24 * 3600

q_sendbuf = np.zeros(shape=(len(recvbuf), len(tf_eval)))
dqdp_sendbuf = np.zeros(shape=(len(recvbuf), 3, len(tf_eval)))

for j, p in enumerate(recvbuf):
    q, dqdp = solve_for_sensitivities(p, 9, x0, tf_eval)
    q_sendbuf[j] = q
    dqdp_sendbuf[j] = dqdp


# everything seems to work up to here :)


### GATHER STEP
q_recvbuf = np.zeros(shape=(sum(count_rows), len(tf_eval)), dtype=np.float64)
dqdp_recvbuf = np.zeros(shape=(sum(count_rows), n_params, len(tf_eval)), dtype=np.float64)

# gather q
if rank == 0:
    count = len(tf_eval) * count_rows
    displ = len(tf_eval) * displ_rows
else:
    count = np.zeros_like(count_rows)
    displ = None
print(f'Rank {rank}; Count {count}; Displ {displ}')
comm.Gatherv(q_sendbuf, [q_recvbuf, count, displ, MPI.DOUBLE], root=0)

# gather dq/dp
if rank == 0:
    count = len(tf_eval) * n_params * count_rows
    displ = len(tf_eval) * n_params * displ_rows
else:
    count = np.zeros_like(count_rows)
    displ = None
comm.Gatherv(dqdp_sendbuf, [dqdp_recvbuf, count, displ, MPI.DOUBLE], root=0)

# finish on the master process
if rank == 0:
    
    print('Gather complete (rank {rank})')
    # save outputs
    np.save('output_q.npy', q_recvbuf)
    np.save('output_dqdp.npy', dqdp_recvbuf)

