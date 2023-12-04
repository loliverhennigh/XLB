from mpi4py import MPI
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        # Sending process
        data_to_send = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        comm.Send(data_to_send, dest=1, tag=11)
        print(f"Process {rank} sent data: {data_to_send}")

    elif rank == 1:
        # Receiving process with a pre-allocated numpy array
        data_to_receive = np.empty(3, dtype=np.float64)
        comm.Recv(data_to_receive, source=0, tag=11)
        print(f"Process {rank} received data: {data_to_receive}")

if __name__ == "__main__":
    main()
