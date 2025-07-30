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

start_time = MPI.Wtime()

### BROADCAST STEP 

if rank == 0:

    # change block to eventually do this smarter; this is fine for now
    # we can open the file, compute the size and the header byte offset in the same with open() block.
    header_size = np.array(128, dtype=int)
    file = np.load(filename, mmap_mode='r')
    file_size = np.array(file.shape)

    # compute the size of each subtask
    ave, res = divmod(file_size[0], size)
    count_rows = [ave + 1 if p < res else ave for p in range(size)]
    count_rows = np.array(count_rows, dtype=int) # number of floats in each array
    print(f'(Rank {rank}) {count_rows}')

else:
    count_rows = np.empty(size, dtype=int)
    file_size = np.empty(2, dtype=int)
    header_size = np.empty(1, dtype=int)

# broadcast count so that every process knows how much data it receives
comm.Bcast(count_rows, root=0)
comm.Bcast(file_size, root=0)
comm.Bcast(header_size, root=0)
n_rows = file_size[0]
n_params = file_size[1]

# compute displ rows
displ_rows = [sum(count_rows[:p]) for p in range(size)]
displ_rows = np.array(displ_rows, dtype=int) # array index to start at

# instantiate buffer matrix; this is where data will be read to
buffer = np.empty(shape=(count_rows[rank], n_params), dtype=np.float64)

# MPI read
fh = MPI.File.Open(comm, filename, MPI.MODE_RDONLY)
offset = (displ_rows[rank] * n_params * MPI.DOUBLE.Get_size()).item()
fh.Read_at_all(header_size + offset, buffer)
fh.Close()
# parameter data per process is saved in buffer for processing.


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

q_sendbuf = np.zeros(shape=(len(buffer), len(tf_eval)))
dqdp_sendbuf = np.zeros(shape=(len(buffer), n_params, len(tf_eval)))

for j, p in enumerate(buffer):
    if (p <= 0).any():
        print(f'(Rank {rank}) ERROR: parameter set p={p} is at index {j} (Global index {displ})')
    q, dqdp = solve_for_sensitivities(p, int(sys.argv[2]), x0, tf_eval)
    q_sendbuf[j] = q
    dqdp_sendbuf[j] = dqdp


### GATHER STEP

# Attempt at using MPI write (epic fail)

# # create the file
# if rank == 0:
#     q_file_shape = (sum(count_rows), len(tf_eval))
#     dqdp_file_shape = (sum(count_rows), n_params, len(tf_eval))
#     q_template = np.empty(shape=q_file_shape, dtype=np.float64)
#     dqdp_template = np.empty(shape=dqdp_file_shape, dtype=np.float64)
#     np.save(f'output_q_{sys.argv[2]}.npy', q_template)
#     np.save(f'output_dqdp_{sys.argv[2]}.npy', dqdp_template)

# # make sure file is created before we proceed with writing
# comm.Barrier()

# # parallel save q
# q_file = MPI.File.Open(comm, f'output_q_{sys.argv[2]}.npy', MPI.MODE_WRONLY)
# offset = (displ_rows[rank] * len(tf_eval) * MPI.DOUBLE.Get_size()).item()
# q_file.Write_at_all(header_size + offset, q_sendbuf)
# q_file.Close()

# # parallel save dqdp
# dqdp_file = MPI.File.Open(comm, f'output_q_{sys.argv[2]}.npy', MPI.MODE_WRONLY)
# offset = (displ_rows[rank] * len(tf_eval) * n_params * MPI.DOUBLE.Get_size()).item()
# dqdp_file.Write_at_all(header_size + offset, dqdp_sendbuf)
# dqdp_file.Close()

# Gather and save on one node (easier and it works)

# gather q
if rank == 0:
    q_recvbuf = np.zeros(shape=(sum(count_rows), len(tf_eval)), dtype=np.float64)
    count = len(tf_eval) * count_rows
    displ = len(tf_eval) * displ_rows
else:
    q_recvbuf = None
    count = None
    displ = None
comm.Gatherv(q_sendbuf, [q_recvbuf, count, displ, MPI.DOUBLE], root=0)

# gather dq/dp
if rank == 0:
    dqdp_recvbuf = np.zeros(shape=(sum(count_rows), n_params, len(tf_eval)), dtype=np.float64)
    count = len(tf_eval) * n_params * count_rows
    displ = len(tf_eval) * n_params * displ_rows
else:
    dqdp_recvbuf = None
    count = None
    displ = None
comm.Gatherv(dqdp_sendbuf, [dqdp_recvbuf, count, displ, MPI.DOUBLE], root=0)

# finish on the master process; save outputs
if rank == 0:
    np.save(f'output_q_{sys.argv[2]}.npy', q_recvbuf)
    np.save(f'output_dqdp_{sys.argv[2]}.npy', dqdp_recvbuf)

end_time = MPI.Wtime()

print(f'(Rank {rank}) Elapsed time: {end_time-start_time:.2f} seconds')

