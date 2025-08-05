/************************************************************************************//**
 * @file definitions.c
 * @author Ján Maťufka - xmatuf00@stud.fit.vutbr.cz
 * @brief Definitions for help functions used in proj2.c
 *
 * Solution to the modified Faneuil Hall problem.
 * Process Synchronization using Semaphores and Shared Memory in C.
 * 2nd project for IOS (Operating Systems) Course - 1BIT FIT VUT - 2020
 * 
 * @date 2020-05-06
 ***************************************************************************************/

#include "definitions.h"

/****************************************************************************************
 *
 * INITIALIZING GLOBAL VARIABLES AND SHARED MEMORY
 *
 ***************************************************************************************/
// initializing global variables for command line arguments
int PI = 0;
int IG = 0;
int JG = 0;
int IT = 0;
int JT = 0;

// shared variables
FILE *file_ptr;
int *log_counter = NULL;
int *NE = NULL;
int *NC = NULL;
int *NB = NULL;
bool *judge_is_inside = NULL;
bool *fork_error = NULL;

// semaphores
sem_t *sem_log_guard = NULL;
sem_t *sem_no_judge_inside = NULL;
sem_t *sem_check_in_queue = NULL;
sem_t *sem_all_checked_in = NULL;
sem_t *sem_judge_confirmed = NULL;
sem_t *sem_children = NULL;

/****************************************************************************************
 *
 * FUNCTION DEFINITINOS
 *
 ***************************************************************************************/
extern void rand_sleep(int miliseconds)
{
	if (miliseconds != 0) 
		usleep((rand() % miliseconds) * 1000);
	return;
} // rand_sleep()


extern void printhelp() 
{
	fprintf(stderr, "Correct usage: ./proj2 PI IG JG IT JT\n");
	fprintf(stderr, "--- PI = ammount of immigrants (>= 1)\n");
	fprintf(stderr, "--- IG = max timeout between generating immigrant processes <0,2000>\n");
	fprintf(stderr, "--- JG = max timeout until judge enters the hall <0,2000>\n");
	fprintf(stderr, "--- IT = max time for getting the certificate <0,2000>\n");
	fprintf(stderr, "--- JT = max time for the confirmation process <0,2000>\n");
	fprintf(stderr, "--- Note: IG JG IT JT are times in miliseconds (ms).\n");
} // print_help()


extern int parse_arguments(int argument_count, char **arguments) 
{
	if (argument_count < 6) return 1;
	char *rest;
	
	PI = (int) strtol(arguments[1], &rest, 10);
	if (errno != 0 || *rest != '\0' || PI < 1 ) return 1;
	
	IG = (int) strtol(arguments[2], &rest, 10);
	if (errno != 0 || *rest != '\0' || IG > 2000 || IG < 0) return 1;

	JG = (int) strtol(arguments[3], &rest, 10);
	if (errno != 0 || *rest != '\0' || JG > 2000 || JG < 0) return 1;

	IT = (int) strtol(arguments[4], &rest, 10);
	if (errno != 0 || *rest != '\0' || IT > 2000 || IT < 0) return 1;

	JT = (int) strtol(arguments[5], &rest, 10);
	if (errno != 0 || *rest != '\0' || JT > 2000 || JT < 0) return 1;

	return 0;
} // parse_arguments()


extern int create_resources()
{
	// opening file for output
	file_ptr = fopen("proj2.out", "w");
	if (file_ptr == NULL) {
		fprintf(stderr, "Could not open/create output file.\n");
		return 1;
	}
	setbuf(file_ptr, NULL);

	// creating shared variables
	MAP_MEMORY(log_counter); 
	MAP_MEMORY(NE);
	MAP_MEMORY(NC);
	MAP_MEMORY(NB);
	MAP_MEMORY(judge_is_inside); 
	MAP_MEMORY(fork_error); 
	
	// initializing shared variables
	*log_counter = 1; 
	*NE = 0;
	*NC = 0;
	*NB = 0;
	*judge_is_inside = false;
	*fork_error = false;
	
	// initializing semaphores
	sem_log_guard = sem_open(SEM_LOG_GUARD_NAME, O_CREAT | O_EXCL, 0666, 1);
	if (sem_log_guard == SEM_FAILED) return 1;

	sem_no_judge_inside = sem_open(SEM_NO_JUDGE_INSIDE_NAME, O_CREAT | O_EXCL, 0666, 1);
	if (sem_no_judge_inside == SEM_FAILED) return 1;

	sem_check_in_queue = sem_open(SEM_CHECK_IN_QUEUE_NAME, O_CREAT | O_EXCL, 0666, 1);
	if (sem_check_in_queue == SEM_FAILED) return 1;

	sem_all_checked_in = sem_open(SEM_ALL_CHECKED_IN_NAME, O_CREAT | O_EXCL, 0666, 0);
	if (sem_all_checked_in == SEM_FAILED) return 1;

	sem_judge_confirmed = sem_open(SEM_JUDGE_CONFIRMED_NAME, O_CREAT | O_EXCL, 0666, 0);
	if (sem_judge_confirmed == SEM_FAILED) return 1;
	
	sem_children = sem_open(SEM_CHILDREN_NAME, O_CREAT | O_EXCL, 0666, 0);
	if (sem_children == SEM_FAILED) return 1;

	// seed random number generator using actual time
	srand(time(0));

	return 0;
} // create_resources()


extern void clean_resources()
{
	// destroying shared variables
	UNMAP_MEMORY(log_counter);
	UNMAP_MEMORY(NE);
	UNMAP_MEMORY(NC);
	UNMAP_MEMORY(NB);
	UNMAP_MEMORY(judge_is_inside);
	UNMAP_MEMORY(fork_error);

	// destroying semaphores
	sem_close(sem_log_guard);
	sem_unlink(SEM_LOG_GUARD_NAME);

	sem_close(sem_no_judge_inside);
	sem_unlink(SEM_NO_JUDGE_INSIDE_NAME);

	sem_close(sem_check_in_queue);
	sem_unlink(SEM_CHECK_IN_QUEUE_NAME);

	sem_close(sem_all_checked_in);
	sem_unlink(SEM_ALL_CHECKED_IN_NAME);

	sem_close(sem_judge_confirmed);
	sem_unlink(SEM_JUDGE_CONFIRMED_NAME);

	sem_close(sem_children);
	sem_unlink(SEM_CHILDREN_NAME);
	
	// closing output file
	fclose(file_ptr);
} // clean_resources()

/*** End of file definitions.c ***/