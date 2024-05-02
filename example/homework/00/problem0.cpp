#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Split processes by rank divided by Q in the world comm
    if (size % P != 0){
        cout << "P must evenly divide the number of processes" << endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Comm color_comm, mod_comm;
    MPI_Comm_split(comm, rank / Q, rank, &color_comm);
    // Split processes by rank % Q in the world comm
    MPI_Comm_split(comm, rank % Q, rank, &mod_comm);
    
    // Build a 2D process topology (P * Q)
    int color_rank, color_size, mod_rank, mod_size;
    MPI_Comm_rank(color_comm, &color_rank);
    MPI_Comm_size(color_comm, &color_size);
    MPI_Comm_rank(mod_comm, &mod_rank);
    MPI_Comm_size(mod_comm, &mod_size);

    // Debugging
    //cout << "Color Rank " << color_rank << " and mod rank " << mod_rank << " for Rank " << rank << endl;

    MPI_Finalize();
    return 0;
}
