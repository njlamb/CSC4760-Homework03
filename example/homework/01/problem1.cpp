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

    int P = 2, Q = size / P;
    if (size % P != 0){
        cout << "P must evenly divide the number of processes" << endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Comm color_comm, mod_comm;
    MPI_Comm_split(comm, rank / Q, rank, &color_comm);
    MPI_Comm_split(comm, rank % Q, rank, &mod_comm);

    int color_rank, color_size, mod_rank, mod_size;
    MPI_Comm_rank(color_comm, &color_rank);
    MPI_Comm_size(color_comm, &color_size);
    MPI_Comm_rank(mod_comm, &mod_rank);
    MPI_Comm_size(mod_comm, &mod_size);

    // Build a 2D topology
    // Using color_comm and mod_comm

    int M = 25;
    vector<int> x(M);
    if (rank == 0){
        for (int i = 0; i < M; i++){
            x[i] = i + 1;
        }
    }

    int* sendcounts = new int[P];
    int* displs = new int[P];

    for (int i = 0; i < P; i++){
        sendcounts[i] = M / P;
        displs[i] = i * M / P;
    }

    // Scatter vector x vertically
    vector<int> x_local(M / P);
    MPI_Scatterv(x.data(), sendcounts, displs, MPI_INT, x_local.data(), M / P, MPI_INT, 0, color_comm);

    delete[] sendcounts;
    delete[] displs;

    // Broadcast vector horizontally
    MPI_Bcast(x_local.data(), M / P, MPI_INT, 0, mod_comm);

    vector<int> y(M / P);

    // Perform parallel copy y := x
    MPI_Allreduce(x_local.data(), y.data(), M / P, MPI_INT, MPI_SUM, mod_comm);

    // Print debug information
    cout << "Rank " << rank << " has y = ";
    for (int i = 0; i < M / P; i++){
        cout << y[i] << " ";
    }
    cout << endl;
    
    MPI_Finalize();
    return 0;
}
