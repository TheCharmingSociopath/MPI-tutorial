#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"

using namespace std;

double find_sum(int low, int high)
{
    double ret = 0;
    for (int i = low; i <= high; ++i) {
        ret += 1.0 / (i * i);
    }
    return ret;
}

int main(int argc, char** argv)
{

    int rank, num_procs, n, ierr;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double start_time = MPI_Wtime();

    // Root process
    if (rank == 0) {
        FILE* file = fopen(argv[1], "r+");
        fscanf(file, "%d", &n);
        fclose(file);

        int elements_per_proc = n / num_procs, rem_elements = n % num_procs, low = 1, high = elements_per_proc;

        for (int i = 1; i < num_procs; ++i) {
            ierr = MPI_Send(&low, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            ierr = MPI_Send(&high, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            low += elements_per_proc;
            high += elements_per_proc;
        }

        double res, temp;

        for (int i = 1; i < num_procs; ++i) {
            ierr = MPI_Recv( &temp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            res += temp;
        }
        res += find_sum(low, high);

        double elapsed_time = MPI_Wtime() - start_time, max_time;
        printf("Total time (s): %f\n", elapsed_time);
        file = fopen(argv[2], "w+");
        fprintf(file, "%f \n", res);
        fclose(file);
    }

    // Sub processes
    else {
        int low, high;
        ierr = MPI_Recv( &low, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        ierr = MPI_Recv( &high, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double res = find_sum(low, high);
        ierr = MPI_Send( &res, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();

    return 0;
}
