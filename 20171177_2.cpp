#include "mpi.h"
#include "algorithm"
#include "stdio.h"
#include "stdlib.h"
#include "vector"

using namespace std;
typedef long long LL;

LL find_pivot(LL* arr, LL low, LL high)
{
    LL pivot = arr[high];
    LL i = low - 1;
    for (LL j = low; j <= high; j++) {
        if (arr[j] < pivot) {
            swap(arr[++i], arr[j]);
        }
    }
    swap(arr[++i], arr[high]);
    return i;
}

void quicksort(LL* arr, LL low, LL high)
{
    if (low < high) {
        LL pivot = find_pivot(arr, low, high);
        quicksort(arr, low, pivot - 1);
        quicksort(arr, pivot + 1, high);
    }
}

int main(int argc, char** argv)
{
    int rank, num_procs;
    vector<LL> arr;

    LL num_elements, temp = 0;

    FILE* file = fopen(argv[1], "r+");
    fscanf(file, "%lld", &num_elements);
    for (int i = 0; i < num_elements; ++i) {
        fscanf(file, "%lld", &temp);
        arr.push_back(temp);
    }
    fclose(file);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    MPI_Bcast(&num_elements, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    LL elements_per_proc = num_elements / num_procs, rem_elements = num_elements % num_procs;
    LL* sub_arr = (LL*)malloc(sizeof(LL) * elements_per_proc);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatter(&arr[rem_elements], elements_per_proc, MPI_LONG_LONG, sub_arr, elements_per_proc, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    quicksort(sub_arr, 0, elements_per_proc - 1);
    MPI_Barrier(MPI_COMM_WORLD);

    // Root process
    if (rank == 0) {
        LL count = rem_elements + elements_per_proc;
        LL *temp_arr = (LL*)malloc(sizeof(LL) * elements_per_proc),
           *final_arr = (LL*)malloc(sizeof(LL) * count);
        for (int i = 0; i < rem_elements; ++i) {
            temp_arr[i] = arr[i];
        }
        quicksort(temp_arr, 0, rem_elements - 1);
        merge(temp_arr, temp_arr + rem_elements, sub_arr, sub_arr + elements_per_proc, final_arr);
        sub_arr = final_arr;

        for (int i = 1; i < num_procs; ++i) {
            count += elements_per_proc;
            final_arr = (LL*)malloc(sizeof(LL) * count);
            MPI_Recv(temp_arr, elements_per_proc, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            merge(sub_arr, sub_arr + count - elements_per_proc, temp_arr, temp_arr + elements_per_proc, final_arr);
            sub_arr = final_arr;
        }
    }

    // Sub processes
    else {
        MPI_Send(sub_arr, elements_per_proc, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_time = MPI_Wtime() - start_time, max_time;
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total time (s): %f\n", max_time);
        FILE* file = fopen(argv[2], "w+");
        for (int i = 0; i < num_elements; ++i) {
            fprintf(file, "%lld ", sub_arr[i]);
        }
        fprintf(file, "\n");
        fclose(file);
    }
    MPI_Finalize();

    return 0;
}
