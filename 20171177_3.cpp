#include "mpi.h"
#include "queue"
#include "stdio.h"
#include "stdlib.h"
#include "vector"

#define MAXX 10000000

using namespace std;

int max(int& a, int& b)
{
    if (a > b) {
        return a;
    }
    return b;
}

int main(int argc, char** argv)
{

    int rank, num_procs, n, m, ierr;

    vector<int> adj[550];
    int colors[550];
    vector<pair<int, int> > edges;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Status status;
    double start_time = MPI_Wtime();

    // Root process
    if (rank == 0) {
        FILE* file = fopen(argv[1], "r+");
        fscanf(file, "%d %d", &n, &m);
        for (int i = 0, a, b; i < m; ++i) {
            fscanf(file, "%d %d", &a, &b);
            --a, --b;
            edges.push_back(make_pair(min(a, b), max(a, b)));
        }
        fclose(file);

        int max_degree = 0;

        for (int i = 0; i < edges.size(); ++i) {
            for (int j = 0; j < edges.size(); ++j) {
                if (edges[i].first == edges[j].first or edges[i].second == edges[j].first
                    or edges[i].first == edges[j].second or edges[i].second == edges[j].second) {
                    adj[i].push_back(j);
                    adj[j].push_back(i);
                    int t1 = adj[i].size(), t2 = adj[j].size();
                    max_degree = max(max_degree, t1);
                    max_degree = max(max_degree, t2);
                }
            }
        }

        int elements_per_proc = m / num_procs, rem_elements = m % num_procs, low = 0, high = elements_per_proc - 1;

        for (int i = 1; i < num_procs; ++i) {
            ierr = MPI_Send(&elements_per_proc, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            ierr = MPI_Send(&low, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            ierr = MPI_Send(&high, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            // ierr = MPI_Send(&colors[low], elements_per_proc, MPI_INT, i, 0, MPI_COMM_WORLD);
            low += elements_per_proc;
            high += elements_per_proc;
        }
        // initialize_colors(colors, elements_per_proc, low, m - 1);
        for (int i = low; i < m; ++i) {
            colors[i] = i;
        }

        for (int i = 1; i < num_procs; ++i) {
            int* temp_arr = (int*)malloc(sizeof(int) * (elements_per_proc + 10));
            ierr = MPI_Recv(temp_arr, elements_per_proc, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            int idx = elements_per_proc * (i - 1);
            for (int j = 0; j < elements_per_proc; ++j) {
                colors[idx + j] = temp_arr[j];
            }
        }
        int mn = MAXX, col;
        for (int i = max_degree + 1; i < m; ++i) {
            vector<int> bucket(max_degree + 5, 1);
            for (int j = 0; j < adj[i].size(); ++j) {
                col = colors[adj[i][j]];
                if (col <= max_degree) {
                    bucket[col] = 0;
                }
            }
            for (int j = 0; j < adj[i].size(); ++j) {
                if (bucket[j] == 1) {
                    colors[i] = j;
                    break;
                }
            }
        }
        double elapsed_time = MPI_Wtime() - start_time, max_time;
        printf("Total time (s): %f\n", elapsed_time);
        file = fopen(argv[2], "w+");
        fprintf(file, "%d \n", max_degree);
        for (int i = 0; i < m; ++i) {
            fprintf(file, "%d ", colors[i]);
        }
        fprintf(file, "\n");
        fclose(file);
    }

    // Sub processes
    else {

        int elements_per_proc, low, high;
        ierr = MPI_Recv(&elements_per_proc, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        ierr = MPI_Recv(&low, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        ierr = MPI_Recv(&high, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        int temp_col[elements_per_proc];
        // ierr = MPI_Recv(&temp_col, elements_per_proc, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        // initialize_colors(temp_col, elements_per_proc, low, high);
        for (int i = 0; i < elements_per_proc; ++i) {
            temp_col[i] = low + i;
        }
        MPI_Send(temp_col, elements_per_proc, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();

    return 0;
}
