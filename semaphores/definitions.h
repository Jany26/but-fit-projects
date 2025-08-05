/************************************************************************************//**
 * @file definitions.h
 * @author Ján Maťufka - xmatuf00@stud.fit.vutbr.cz
 * @brief Declarations of global/shared variables, function prototypes + documentation.
 *
 * Solution to the modified Faneuil Hall problem.
 * Process Synchronization using Semaphores and Shared Memory in C.
 * 2nd project for IOS (Operating Systems) Course - 1BIT FIT VUT - 2020
 * 
 * @date 2020-05-06
 ***************************************************************************************/

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

/****************************************************************************************
 *
 * LIBRARIES NEEDED
 *
 ***************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>  
#include <time.h>

#include <fcntl.h>
#include <unistd.h>

#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <semaphore.h>

/****************************************************************************************
 *
 * MACROS / DEFINES
 *
 ***************************************************************************************/

// creates shared memory for variable stored at the address pointed to by ptr
#define MAP_MEMORY(pointer) ((pointer) = mmap(NULL, sizeof(*(pointer)), \
	PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0)) 

// unmaps shared memory at ptr adress
#define UNMAP_MEMORY(pointer) (munmap((pointer), sizeof(pointer)))

// semaphore names
#define SEM_LOG_GUARD_NAME "/xmatuf00.ios-proj2.sem_log_guard"
#define SEM_NO_JUDGE_INSIDE_NAME "/xmatuf00.ios-proj2.sem_no_judge_inside"
#define SEM_CHECK_IN_QUEUE_NAME "/xmatuf00.ios-proj2.sem_check_in_queue"
#define SEM_ALL_CHECKED_IN_NAME "/xmatuf00.ios-proj2.sem_all_checked_in"
#define SEM_JUDGE_CONFIRMED_NAME "/xmatuf00.ios-proj2.sem_judge_confirmed"
#define SEM_CHILDREN_NAME "/xmatuf00.ios-proj2.sem_children"

/****************************************************************************************
 *
 * DECLARING GLOBAL (& SHARED) VARIABLES
 *
 ***************************************************************************************/

// command line arguments
extern int PI; /**< PI = amount of immigrant processes */ 
extern int IG; /**< IG = max timeout for generating immigrant process */
extern int JG; /**< JG = max timeout before judge enters the hall */
extern int IT; /**< IT = max timeout until immigrant gets a certificate */
extern int JT; /**< JT = max time for confirmation process */

// shared variables
extern FILE *file_ptr;
extern int *log_counter;   /**< for proper printing to the log file */
extern int *NE;            /**< amount of immigrants that entered the building */
extern int *NC;            /**< amount of immigrants that have checked */
extern int *NB;            /**< amount of immigrants that are still in the building */
extern bool *judge_is_inside; /**< whether or not the judge is inside the building */
extern bool *fork_error;   /**< for handling errors while generating processes */

// semaphores
extern sem_t *sem_log_guard;        /**< guarding access to logging file (and action counter) */
extern sem_t *sem_no_judge_inside;  /**< waiting for judge to leave the building, also protects entering */
extern sem_t *sem_check_in_queue;   /**< guarding the check in */
extern sem_t *sem_all_checked_in;   /**< guards the start to confirmation, until all conditions are met */
extern sem_t *sem_judge_confirmed;  /**< guarding access to the hall during confirmation process */
extern sem_t *sem_children;

/****************************************************************************************
 *
 * FUNCTION PROTOTYPES AND DOCUMENTATION
 *
 ***************************************************************************************/

/************************************************************************************//**
 *
 * @brief Loads all arguments into global variables.
 * @param argument_count argc (amount of CLI arguments)
 * @param arguments argv (array of CLI argument strings)
 * @return 1 if an error is encountered (not enough arguments, invalid arguments, out of bounds values)
 * @return 0 if arguments are loaded successfully
 *
 ***************************************************************************************/
extern int parse_arguments(int argument_count, char **arguments);

/************************************************************************************//**
 *
 * @brief Put the process to sleep for random amound of miliseconds.
 * @param milisecnods Max amount of time to sleep. (for this program max 2000)
 *
 ***************************************************************************************/
extern void rand_sleep(int miliseconds);

/************************************************************************************//**
 *
 * @brief Prints help (how to use this program) when wrong arguments are entered.
 *
 ***************************************************************************************/
extern void printhelp();

/************************************************************************************//**
 *
 * @brief Initializes all shared variables, semaphores and file for output, seeds the RNG.
 *
 ***************************************************************************************/
extern int create_resources();

/************************************************************************************//**
 *
 * @brief Unmaps shared variables, destroys semaphores, closes file.
 *
 ***************************************************************************************/
extern void clean_resources();

#endif // DEFINITIONS_H
/*** End of file definitions.h ***/