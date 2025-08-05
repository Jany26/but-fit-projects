/**
 * @file life.cpp
 * @author Ján Maťufka (xmatuf00@stud.fit.vutbr.cz)
 * @brief Game of life implementation using MPI.
 * 2nd project of PRL (Parallel and distributed algorithms) course.
 * @date 2024-04-21
 */

#include <fstream>
#include <vector>
#include <mpi.h>

/* if WRAP_AROUND is set to true, then first column neighbors the last column
 * and top row neighbors the bottom row
 * if false, then every cell out of bounds is considered dead
 */
#define WRAP_AROUND true 

#define TAG_OK 0
#define TAG_ERR 1


/**
 * @brief Encapsulates all important information for 1 process.
 * Each process handles 1 row of the game.
 */
typedef struct {
    int rank;       // process ID (determines the position in the pipeline)
    int rank_count; // how many processes in total are there
    std::vector<char> cells; // game cell representation
} process_properties;


/**
 * @brief Initialize structure with the properties of the process.
 * @return process_properties structure with the initialized data
 */
process_properties initialize_process() {
    process_properties proc;
    MPI_Comm_size(MPI_COMM_WORLD, &proc.rank_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc.rank);
    proc.cells = {};
    return proc;
}


/**
 * @brief Print out the content of the buffer (for debugging).
 * 
 * @param buf vector representing one row of game cells
 */
void print_buffer_content(std::vector<char> buf) {
    printf("[");
    for (int i = 0; i < buf.size(); i++) {
        printf(" %d", buf[i]);
    }
    printf(" ]\n");
}


/**
 * @brief Read file and send game rows to corresponding processes.
 * 
 * @param proc Process that reads the file (rank = 0).
 * @param filename Path to the file with game representation.
 * @return int Number of valid lines of the game matrix if successful.
 * @return -1 If an error is encountered (inconsistent line lengths, invalid characters).
 */
int stupid_read_file(process_properties *proc, char *filename) {
    std::vector<char> buffer;
    int i = 0;
    int first_line_size = 0;
    int current_line = 0;
    int result = 0;
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "read_file: cannot open file\n");
        // other processes expect 3 messages, we send them with ERR TAG for correct termination
        for (int i = 1; i < proc->rank_count; i++) {
            MPI_Send(&current_line, 1, MPI_INT, i, TAG_ERR, MPI_COMM_WORLD);
            MPI_Send(&current_line, 0, MPI_CHAR, i, TAG_ERR, MPI_COMM_WORLD);
            MPI_Send(&current_line, 1, MPI_INT, i, TAG_ERR, MPI_COMM_WORLD);
        }
        return -1;
    }
    int c;
    while ((c = fgetc(file)) != EOF) {
        switch(c) {
            case '\n':
                if (current_line == 0) {
                    first_line_size = i;
                    MPI_Status stat;
                    proc->cells.resize(i);
                    // we perform the copy by sending and receiving 
                    MPI_Send(buffer.data(), i, MPI_CHAR, 0, TAG_OK, MPI_COMM_WORLD);
                    MPI_Recv(proc->cells.data(), i, MPI_CHAR, 0, TAG_OK, MPI_COMM_WORLD, &stat);
                } else {
                    // the lines need to have the same number of cells
                    if (first_line_size != i) {
                        fprintf(stderr, "read_file: line %i: inconsistent number of cells (%i)\n", current_line + 1, i);
                        result = -1;
                    }
                    MPI_Send(&i, 1, MPI_INT, current_line, TAG_OK, MPI_COMM_WORLD);
                    MPI_Send(buffer.data(), i, MPI_CHAR, current_line, TAG_OK, MPI_COMM_WORLD);
                }
                current_line++;
                buffer.clear();
                i = 0;
                break;
            case '0':
                buffer.push_back(0);
                i++;
                break;
            case '1':
                buffer.push_back(1);
                i++;
                break;
            default: // invalid character
                fprintf(stderr, "read_file: line %d: skipped invalid character '%c'\n", current_line, c);
                result = -1;
                break;
        }
    }
    for (int i = 1; i < proc->rank_count; i++) {
        MPI_Send(&current_line, 1, MPI_INT, i, (result != -1) ? TAG_OK : TAG_ERR, MPI_COMM_WORLD);
    }
    return (result != -1) ? current_line : result;
}


/**
 * @brief Compute the next iteration row of game cells using neigboring rows (wrap-around playing field).
 * 
 * @param proc Process that performs the computing (has the current iteration stored in its inner vector).
 * @param top Row of cells neighboring the currently computed row from the top.
 * @param bot Row of cells neighboring the currently computed row from the bottom.
 * @param result Temporary vector for storing the computed result (will be copied to proc->cells).
 */
void compute_wraparound(process_properties *proc, char *top, char *bot, char *result) {
    int alive_cells, left, right;
    for (int i = 0; i < proc->cells.size(); i++) {
        left = (i == 0) ? proc->cells.size() - 1 : i - 1;
        right = (i == proc->cells.size() - 1) ? 0 : i + 1;
        alive_cells =  top[left] + top[i] + top[right];
        alive_cells += (proc->cells[left] + proc->cells[right]);
        alive_cells += (bot[left] + bot[i] + bot[right]);
        if (proc->cells[i] && (alive_cells < 2 || alive_cells > 3)) {
            result[i] = 0;
        } else if (!proc->cells[i] && alive_cells == 3) {
            result[i] = 1;
        } else {
            result[i] = proc->cells[i];
        }
    }
}


/**
 * @brief Compute the next iteration row of game cells using neigboring rows (dead boundaries around the playing field).
 * 
 * @param proc Process that performs the computing (has the current iteration stored in its inner vector).
 * @param top Row of cells neighboring the currently computed row from the top.
 * @param bot Row of cells neighboring the currently computed row from the bottom.
 * @param result Temporary vector for storing the computed result (will be copied to proc->cells).
 */

void compute_hardwalls(process_properties *proc, char *top, char *bot, char *result) {
    int alive_cells;
    for (int i = 0; i < proc->cells.size(); i++) {
        alive_cells = top[i] + bot[i];
        if (i != 0) {
            alive_cells += top[i-1] + proc->cells[i-1] + bot[i-1];
        }
        if (i != proc->cells.size() - 1) {
            alive_cells += (top[i+1] + proc->cells[i+1] + bot[i+1]);
        }
        if (proc->cells[i] && (alive_cells < 2 || alive_cells > 3)) {
            result[i] = 0;
        } else if (!proc->cells[i] && alive_cells == 3) {
            result[i] = 1;
        } else {
            result[i] = proc->cells[i];
        }
    }
}


/**
 * @brief Compute next game iteration of cell states for one row (one process).
 * 
 * @param proc Process that computes the next iteration states for cells stored inside.
 */
void game_iteration(process_properties *proc) {
    // by default initialized with 0s (simulating outside of the game field)
    // only filled with cell data when appropriate
    std::vector<char> top_row(proc->cells.size(), 0);
    std::vector<char> bottom_row(proc->cells.size(), 0);
    // A) send phase
    // any row but top can freely send cells up
    if (proc->rank != 0) {
        MPI_Send(proc->cells.data(), proc->cells.size(), MPI_CHAR, proc->rank - 1, TAG_OK, MPI_COMM_WORLD);
    }
    // top row can only send to the last process/row only when wraparound is used
    if (proc->rank == 0 && WRAP_AROUND) {
        MPI_Send(proc->cells.data(), proc->cells.size(), MPI_CHAR, proc->rank_count - 1, TAG_OK, MPI_COMM_WORLD);
    }
    // any row but bottom can send cells down
    if (proc->rank != proc->rank_count - 1) {
        MPI_Send(proc->cells.data(), proc->cells.size(), MPI_CHAR, proc->rank + 1, TAG_OK, MPI_COMM_WORLD);
    }
    // bottom row can send to the first process/row (index 0) only when wraparound is used
    if (proc->rank == proc->rank_count - 1 && WRAP_AROUND) {
        MPI_Send(proc->cells.data(), proc->cells.size(), MPI_CHAR, 0, TAG_OK, MPI_COMM_WORLD);
    }
    // B) recv phase = to temporary buffers
    // top and bottom processes will not overwrite buffers when wraparound is not used
    MPI_Status status;
    if (proc->rank != proc->rank_count -1) {
        MPI_Recv(bottom_row.data(), proc->cells.size(), MPI_CHAR, proc->rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
    if (proc->rank == proc->rank_count -1 && WRAP_AROUND) {
        MPI_Recv(bottom_row.data(), proc->cells.size(), MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
    if (proc->rank != 0) {
        MPI_Recv(top_row.data(), proc->cells.size(), MPI_CHAR, proc->rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
    if (proc->rank == 0 && WRAP_AROUND) {
        MPI_Recv(top_row.data(), proc->cells.size(), MPI_CHAR, proc->rank_count - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }

    // C) computation phase
    std::vector<char> result(proc->cells.size(), 0);
    if (WRAP_AROUND) {
        compute_wraparound(proc, top_row.data(), bottom_row.data(), result.data());
    } else {
        compute_hardwalls(proc, top_row.data(), bottom_row.data(), result.data());
    }
    // D) rewrite phase
    for (int i = 0; i < proc->cells.size(); i++) {
        proc->cells[i] = result[i];
    }
}


/**
 * @brief Print out the current state of the game.
 * 
 * All processes send their data to process with rank 0.
 * Process with rank 0 prints the output (in order for the rows to be ordered).
 * 
 * @param proc
 */
void print_game_state(process_properties *proc) {
    MPI_Send(proc->cells.data(), proc->cells.size(), MPI_CHAR, 0, TAG_OK, MPI_COMM_WORLD);
    if (proc->rank == 0) {
        for (int j = 0; j < proc->rank_count; j++ ) {
            std::vector<char> buffer(proc->cells.size(), 0);
            MPI_Status status;
            MPI_Recv(buffer.data(), proc->cells.size(), MPI_CHAR, j, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            printf("%d: ", j);
            for (int k = 0; k < proc->cells.size(); k++) {
                printf("%d", buffer[k]);
            }
            printf("\n");
        }
    }
}


/**
 * Program is run with two arguments.
 * 1) file name with the initial configuration of the game
 * 2) number of steps the game has to perform
 */
int main(int argc, char **argv) {
    MPI_Init(&argc,&argv);
    process_properties p = initialize_process();
    process_properties *proc = &p;

    // input reading and game setup
    if (proc->rank == 0) {
        int err_check = stupid_read_file(proc, argv[1]);
        if (err_check == -1) {
            fprintf(stderr, "read_file: failed\n");
            MPI_Finalize();
            return 0;
        }
    } else {
        MPI_Status status;
        int length;
        MPI_Recv(&length, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        proc->cells.resize(length);
        MPI_Recv(proc->cells.data(), length, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&length, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TAG_ERR) {
            MPI_Finalize();
            return 0;
        }
    }
    // main game loop
    for (int i = 0; i < atoi(argv[2]); i++) {
        game_iteration(proc);
    }

    print_game_state(proc);
    MPI_Finalize();
    return 0;
}
