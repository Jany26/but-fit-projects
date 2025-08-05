/**
 * @file pms.cpp
 * @author Ján Maťufka (xmatuf00@stud.fit.vutbr.cz)
 * @brief Pipeline merge-sort implementation using MPI.
 * 1st project of PRL (Parallel and distributed algorithms) course.
 * @date 2024-04-06
 */

#include <fstream>
#include <queue>
#include <mpi.h>

#define DEBUG_PRINT false // set to true for debugging/testing
 
#if DEBUG_PRINT
#define debug_print(fmt, ... ) \
    printf("DEBUG: " fmt,__VA_ARGS__);\
    printf("\n");
#else
#define debug_print(fmt, ...)
#endif

#define INPUT_FILENAME "./numbers" // the input will always be the same

#define CONTINUE 0 // sent with all but the last number from the input
#define STOP 1 // sent with the last number of the input sequence 


/**
 * @brief Convert the content of a queue into string (for debugging).
 * 
 * @param q queue to print
 * @return string representation of the queue: "< 0 1 2 3 ... >"
 */
std::string queue_str(std::queue<int> q) {
    std::string result = "< ";
    for (size_t i = 0; i < q.size(); i++) {
        result += std::to_string(q.front());
        result += " ";
        q.push(q.front());
        q.pop();
    }
    return result + ">";
}

/**
 * @brief Read the input and put it load it into a queue.
 * 
 * @param dest pointer to the queue
 * @return int how many numbers (bytes) were read
 */
int read_numbers(std::queue<int> *dest) {
    int b, count = 0;
    std::fstream instream;
    instream.open(INPUT_FILENAME, std::ios::in);
    for (b = instream.get(); instream.good(); b = instream.get()) {
        count++;
        dest->push((int) b);
        printf("%d ", (int) b);
    }
    printf("\n");
    return count;
} 

/**
 * @brief Contains all necessary (+ extra for debugging) information about the process.
 */
typedef struct {
    int rank;       // process ID (determines the position in the pipeline)
    int rank_count; // how many processes in total are there
    bool is_last;   // last process works slightly different than middle
    int sent_cnt;   // for debugging purposes (debug prints etc.)
    int recv_cnt;   // for debugging purposes (debug prints etc.)
    std::queue<int> up_buf = {};
    std::queue<int> low_buf = {};
    // while we are producing the current batch (i.e. merge two sub-batches from
    // incoming input queues), we need to keep track of:
    int up_cap;     // how many items to send from the upper queue
    int low_cap;    // how many items remain unsent from the lower queue 
} process_properties;

/**
 * @brief Initialize structure with the properties of the process.
 * 
 * @param rank ID of the process
 * @param size how many processes are there in total
 * @return process_properties structure with the initialized data
 */
process_properties initialize_process(int rank, int size) {
    process_properties proc;
    proc.rank = rank;
    proc.rank_count = size;
    proc.is_last = rank == (size - 1);
    proc.sent_cnt = 0;
    proc.recv_cnt = 0;
    proc.up_buf = {};
    proc.low_buf = {};
    proc.up_cap = 1 << (rank - 1);
    proc.low_cap = 1 << (rank - 1);
    return proc;
}

/**
 * @brief Read input and send it (number by number) to the next processor.
 * 
 * Handles the work of the first processor in the pipeline.
 * Note: Last number from the queue is sent with the STOP tag.
 * 
 * @param proc pointer to the structure with process properties
 */
void input_process(process_properties *proc) {
    int input_size = read_numbers(&proc->up_buf);
    int current_value, tag;
    while (proc->sent_cnt < input_size) {
        current_value = proc->up_buf.front();
        proc->sent_cnt++;
        tag = (proc->sent_cnt == input_size) ? STOP : CONTINUE;
        // if we have only 1 number, this processor is also the last one
        if (proc->rank_count <= 1) {
            printf("%d\n", current_value);
        } else {
            MPI_Send(&current_value, 1, MPI_INT, proc->rank + 1, tag, MPI_COMM_WORLD);
        }
        proc->up_buf.pop();
        debug_print("%d - sent %d. value %d from 1 queue, [t=%d] upper = %s, lower = %s",
            proc->rank, proc->sent_cnt, current_value, tag,
            queue_str(proc->up_buf).c_str(), queue_str(proc->low_buf).c_str());
    }
}

/**
 * @brief Check into which queue to put the currently received number.
 * 
 * cut off value = half of the cycle_size 
 * 0    will not use this function, since it does not need to use MPI_Recv()
 * rank cycle_size  cutoff  modulo sequence         resulting sequence
 * 1    2^(1)=2      < 1    0101010101010101...     ULULULULULULULUL...
 * 2    2^(2)=4      < 2    0123012301230123...     UULLUULLUULLUULL...
 * 3    2^(3)=8      < 4    0123456701234567...     UUUULLLLUUUULLLL...
 * ...
 * i    2^(i)=N      < N/2  01234...(N-1)012...     N/2 Us and then N/2 Ls...
 * ...
 * last                                             UUUUUUUUUUUUUUUU...
 * 
 * @param i current iteration / index of the received number
 * @param rank process ID
 * @return true received number should be put into the upper input buffer
 * @return false received number should be put into the upper lower buffer
 */
bool upper_buf(int i, int rank) {
    int cycle_size = 1 << (rank);
    return (i % cycle_size < (cycle_size >> 1)) ? true : false;
}

/**
 * @brief Handle receiving of the number from previous process.
 * 
 * @param proc pointer to the structure with process properties
 * @param tag the tag is set to STOP when the last number was received
 */
void receive_number(process_properties *proc, int *tag) {
    int recv_upper, current_value;
    MPI_Status status;
    recv_upper = upper_buf(proc->recv_cnt, proc->rank);
    MPI_Recv(&current_value, 1, MPI_INT, proc->rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    *tag = status.MPI_TAG;
    (recv_upper ? proc->up_buf : proc->low_buf).push(current_value);
    proc->recv_cnt++;
    debug_print("%d - recv %d. value %d into %d queue, [t=%d] upper = %s, lower = %s",
        proc->rank, proc->recv_cnt, current_value, recv_upper, *tag, 
        queue_str(proc->up_buf).c_str(), queue_str(proc->low_buf).c_str());
}

/**
 * @brief Fill the input queues with the appropriate number of elements.
 * 
 * For seamless pipeline processing:
 * upper queue has to have at least 2^(rank-1) items
 * lower queue has to have at least 1 item
 * OR we jump into merging sooner, because the input sequence is too short
 * (ie. the tag already became STOP)
 * 
 * @param proc pointer to the structure with process properties
 * @param tag for checking whether the whole input was already processed
 * @see process_properties structure definition
 */
void prefill_buffers(process_properties *proc, int *tag) {
    while (!(
        ((int) proc->up_buf.size() >= (1 << (proc->rank - 1)) &&
        (int) proc->low_buf.size() >= 1) || (*tag == STOP)
    )) {
        receive_number(proc, tag);
    }
    debug_print("%d - after prefill %s %s\n", proc->rank, queue_str(proc->up_buf).c_str(), queue_str(proc->low_buf).c_str());
}

/**
 * @brief Choose input queue from which to send the number along the pipeline.
 * 
 * The output batch consists of 2 equal-size sub-batches, except for the
 * last batch in case of input size not equal to a power of two.
 * Each process with rank i merges exactly 2^(i-1) items (within a full batch)
 * from the upper queue with 2^(i-1) items from the lower queue.
 * 
 * @param proc pointer to structure with process_properties
 * @return true if the front of the upper queue is picked
 * @return false if the front of the lower queue is picked
 */
bool pick_queue(process_properties *proc) {
    int u_item = proc->up_buf.front();
    int l_item = proc->low_buf.front();
    bool result;
    if (proc->up_cap && proc->low_cap) {
        if (u_item < l_item) { // SEND UPPER
            debug_print("%d - pick_queue: (%d, %d) [%d, %d] - picking upper (compare) [%d]",
                proc->rank, proc->up_cap, proc->low_cap, u_item, l_item, u_item);
            proc->up_cap--;
            result = true;
        } else { // SEND LOWER
            debug_print("%d - pick_queue: (%d, %d) [%d, %d] - picking lower (compare) [%d]",
                proc->rank, proc->up_cap, proc->low_cap, u_item, l_item, l_item);
            proc->low_cap--;
            result = false;
        }
    } else if (proc->up_cap) { // SEND UPPER
        debug_print("%d - pick_queue: (%d, %d) [%d, %d] - picking upper (capacity) [%d]",
            proc->rank, proc->up_cap, proc->low_cap, u_item, l_item, u_item);
        proc->up_cap--;
        result = true;
    } else if (proc->low_cap) {// SEND LOWER
        debug_print("%d - pick_queue: (%d, %d) [%d, %d] - picking lower (capacity) [%d]",
            proc->rank, proc->up_cap, proc->low_cap, u_item, l_item, l_item);
        proc->low_cap--;
        result = false;
    }
    return result;
}

/**
 * @brief Send a number to the next processor (or output).
 * 
 * @param proc pointer to the structure with process properties
 * @param tag for checking whether the whole input was already processed
 * @see process_properties structure definition
 */
void send_number(process_properties *proc, int *tag, bool end) {
    int send_tag = CONTINUE;
    int send_upper, current_value;
    send_upper = pick_queue(proc);
    current_value = send_upper ? proc->up_buf.front() : proc->low_buf.front();
    (send_upper ? proc->up_buf : proc->low_buf).pop();
    send_tag = (*tag == CONTINUE || !(proc->up_buf.empty() && proc->low_buf.empty())) ? CONTINUE : STOP;

    // last process does not have to send the number, just print it to stdout
    if (proc->rank == proc->rank_count - 1) {
        printf("%d\n", current_value);
    } else {
        MPI_Send(&current_value, 1, MPI_INT, proc->rank + 1, send_tag, MPI_COMM_WORLD);
    }
    proc->sent_cnt++;
    debug_print("%d - sent %d. value %d from %d queue, [t=%d] upper = %s, lower = %s",
        proc->rank, proc->sent_cnt, current_value, send_upper, *tag,
        queue_str(proc->up_buf).c_str(), queue_str(proc->low_buf).c_str());
}

/**
 * @brief Send all numbers after the buffers were appropriatly filled.
 * 
 * The loop is split into batches (for correct splitting of input into batches).
 * 
 * In each iteration of the batch processing, the function keeps receiving
 * a number (if STOP tag was not sent yet) and then sending it to the next proceses.
 * 
 * @param proc pointer to the structure with process properties
 * @param tag for checking whether the whole input was already processed
 * @see process_properties structure definition
 */
void sending_loop(process_properties *proc, int *tag) {
    // a) empty buffer checking is not sufficient, since the input
    //    might not have been produced by a previous process
    // b) tag checking is not sufficient either, since we can already be done
    //    with receiving input but still have unprocessed items in the queues
    bool end = false;
    while (*tag == CONTINUE || !(proc->up_buf.empty() && proc->low_buf.empty())) {
        proc->up_cap = 1 << (proc->rank - 1);
        proc->low_cap = 1 << (proc->rank - 1);
        while (proc->up_cap || proc->low_cap) {
            // tag check here is necessary, otherwise we will be waiting for
            // an item that will never come
            if (*tag != STOP) {
                receive_number(proc, tag);
            }
            // if we have already received all numbers and the expected
            // capacities are higher than the actual number of items in queues,
            // we need to adjust the capacities (because we are processing the
            // last batch which could have a different size than a power of 2)
            if (*tag == STOP && (
                proc->up_cap > (int) proc->up_buf.size() ||
                proc->low_cap > (int) proc->low_buf.size()
                )
            ) {
                proc->up_cap = (int) proc->up_buf.size();
                proc->low_cap = (int) proc->low_buf.size();
            }
            send_number(proc, tag, end);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size == 0) {
        return 1;
    }

    process_properties p = initialize_process(rank, size);
    process_properties *proc = &p;

    int tag = CONTINUE;
    if (proc->rank == 0) {
        input_process(proc);
    } else {
        prefill_buffers(proc, &tag);
        sending_loop(proc, &tag);
    }

    debug_print("%d - done", proc->rank);
    MPI_Finalize();
    return 0;
}
