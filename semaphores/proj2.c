/************************************************************************************//**
 * @file proj2.c
 * @author Ján Maťufka - xmatuf00@stud.fit.vutbr.cz
 * @brief Main function + all processes definitions.
 *
 * Solution to the modified Faneuil Hall problem
 * Process Synchronization using Semaphores and Shared Memory in C
 * 2nd project for IOS (Operating Systems) Course - 1BIT FIT VUT - 2020
 *
 * @date 2020-05-06
 ***************************************************************************************/

#include "definitions.h"
void process_judge()
{
	int total_immigrants_confirmed = 0; // checking for 
	while (total_immigrants_confirmed < PI) {

		rand_sleep(JG);

		// JUDGE WANTS TO ENTER

		sem_wait(sem_log_guard);
		fprintf(file_ptr, "%d\t: JUDGE\t: wants to enter\n", *log_counter);
		(*log_counter)++;	
		sem_post(sem_log_guard);

		sem_wait(sem_no_judge_inside);
		sem_wait(sem_check_in_queue);

		// JUDGE ENTERING

		sem_wait(sem_log_guard);
		fprintf(file_ptr, "%d\t: JUDGE\t: enters\t\t: %d\t: %d\t: %d\n", *log_counter, *NE, *NC, *NB);
		(*log_counter)++;
		*judge_is_inside = true;
		sem_post(sem_log_guard);

		// (Not always) JUDGE WAITING FOR ALL IMMIGRANTS IN THE BUILDING TO CHECK

		if (*NE > *NC) {
			sem_wait(sem_log_guard);
			fprintf(file_ptr, "%d\t: JUDGE\t: waits for imm\t\t: %d\t: %d\t: %d\n", *log_counter, *NE, *NC, *NB);
			(*log_counter)++;
			sem_post(sem_log_guard);

			sem_post(sem_check_in_queue); // allow immigrants to check in
			sem_wait(sem_all_checked_in); // wait for last immigrant to check in
		}

		// JUDGE STARTING CONFIRMATION

		sem_wait(sem_log_guard);
		fprintf(file_ptr, "%d\t: JUDGE\t: starts confirmation\t: %d\t: %d\t: %d\n", *log_counter, *NE, *NC, *NB);
		(*log_counter)++;
		sem_post(sem_log_guard);

		rand_sleep(JT);

		// JUDGE ENDING CONFIRMATION

		sem_wait(sem_log_guard);
		
		int current_immigrants_to_confirm = *NC;
		total_immigrants_confirmed += *NC;

		*NE = 0; // no one is now waiting at entrance
		*NC = 0; // no one is now waiting at check in
		
		for(int i = 0; i < current_immigrants_to_confirm; i++)
			sem_post(sem_judge_confirmed); // signaling to immigrants they can now get their certificates
		
		fprintf(file_ptr, "%d\t: JUDGE\t: ends confirmation\t: %d\t: %d\t: %d\n", *log_counter, *NE, *NC, *NB);
		(*log_counter)++;
		
		sem_post(sem_log_guard);

		rand_sleep(JT);
		
		// JUDGE LEAVING THE BUILDING

		sem_wait(sem_log_guard);
		fprintf(file_ptr, "%d\t: JUDGE\t: leaves\t\t: %d\t: %d\t: %d\n", *log_counter, *NE, *NC, *NB);
		(*log_counter)++;
		*judge_is_inside = false; // judge is no longer inside
		sem_post(sem_log_guard);

		sem_post(sem_check_in_queue); // now allowing registration
		sem_post(sem_no_judge_inside); // now immigrants can go inside the building 
	}

	// WHEN ALL IMMIGRANTS ARE CONFIRMED, JUDGE CAN FINISH

	sem_wait(sem_log_guard);
	fprintf(file_ptr, "%d\t: JUDGE\t: finishes\n", *log_counter);
	(*log_counter)++;
	sem_post(sem_log_guard);

	sem_post(sem_children); // signal main
	exit(0);
}


void process_immigrant(int ID)
{
	// IMMIGRANT STARTING

	sem_wait(sem_log_guard);
	fprintf(file_ptr, "%d\t: IMM %d\t: starts\n", *log_counter, ID); // critical section
	(*log_counter)++;
	sem_post(sem_log_guard);

	// IMMIGRANT ENTERING

	sem_wait(sem_no_judge_inside); // wait for turn to enter
	
	sem_wait(sem_log_guard);
	(*NE)++;
	(*NB)++;
	fprintf(file_ptr, "%d\t: IMM %d\t: enters\t\t: %d\t: %d\t: %d\n", *log_counter, ID, *NE, *NC, *NB);
	(*log_counter)++;
	sem_post(sem_log_guard);

	sem_post(sem_no_judge_inside); // allow next person to enter

	// IMMIGRANT CHECKING IN

	sem_wait(sem_check_in_queue); // wait for turn to check in

	sem_wait(sem_log_guard);
	(*NC)++;
	fprintf(file_ptr, "%d\t: IMM %d\t: checks\t\t: %d\t: %d\t: %d\n", *log_counter, ID, *NE, *NC, *NB);
	(*log_counter)++;
	sem_post(sem_log_guard);

	sem_wait(sem_log_guard); // guard access to NE and NC
	if (*judge_is_inside && *NE == *NC) { // check if everyone is ready for confirmation
		sem_post(sem_all_checked_in); // now the judge can start the confirmation 
	}
	else {
		sem_post(sem_check_in_queue); // otherwise allow next person to check in
	}
	sem_post(sem_log_guard);


	sem_wait(sem_judge_confirmed); // wait for confirmation to end

	// IMMIGRANT WANTS THE CERTIFICATE

	sem_wait(sem_log_guard);
	fprintf(file_ptr, "%d\t: IMM %d\t: wants certificate\t: %d\t: %d\t: %d\n", *log_counter, ID, *NE, *NC, *NB);
	(*log_counter)++;
	sem_post(sem_log_guard);

	rand_sleep(IT);

	// IMMIGRANT GETS THE CERTIFICATE

	sem_wait(sem_log_guard);
	fprintf(file_ptr, "%d\t: IMM %d\t: got certificate\t: %d\t: %d\t: %d\n", *log_counter, ID, *NE, *NC, *NB);
	(*log_counter)++;
	sem_post(sem_log_guard);

	// IMMIGRANT LEAVING

	sem_wait(sem_no_judge_inside); // wait for turn to leave
	
	sem_wait(sem_log_guard);
	(*NB)--;
	fprintf(file_ptr, "%d\t: IMM %d\t: leaves\t\t: %d\t: %d\t: %d\n", *log_counter, ID, *NE, *NC, *NB);
	(*log_counter)++;
	sem_post(sem_log_guard);

	sem_post(sem_no_judge_inside); // allow next person to enter/leave

	sem_post(sem_children); // signal main
	exit(0);
}

void process_immigrant_generator() 
{
	for(int i = 1; i <= PI; i++) {
		rand_sleep(IG);
		pid_t immigrant = fork();
		if (immigrant == 0 && !(*fork_error))
			process_immigrant(i);
		else if (immigrant < 0) {
			fprintf(stderr, "ERROR: GENERATOR failed to create IMMIGRANT %d process.\n", i);
			*fork_error = true;
			clean_resources();
			exit(1);
		}
	}

	sem_post(sem_children); // signal main
	exit(0);
}

int main(int argc, char **argv)
{
	// LOAD ARGUMENTS

	if (parse_arguments(argc, argv) != 0) {
		fprintf(stderr, "ERROR: Invalid arguments.\n");
		printhelp();
		return 1;
	}

	// INITIALIZE

	if (create_resources() != 0) {
		fprintf(stderr, "ERROR: Could not create shared memory, file or semaphores properly.\n");
		clean_resources();
		return 1;
	}

	// CREATE JUDGE

	pid_t judge = fork();
	if (judge == 0 && !(*fork_error)) {
		process_judge();
	} 
	else if (judge < 0) {
		fprintf(stderr, "ERROR: Failed to create JUDGE process.\n");
		*fork_error = true;
		clean_resources();
		return 1;
	}
	
	// CREATE GENERATOR

	pid_t generator = fork();
	if (generator == 0 && !(*fork_error)) {
		process_immigrant_generator();
	}
	else if (generator < 0) {
		fprintf(stderr, "ERROR: Failed to create immigrant GENERATOR process.\n");
		*fork_error = true;
		clean_resources();
		return 1;
	}

	// WAIT FOR ALL CHILDREN PROCESSES TO END

	for (int i = 0; i < (PI + 2); i++) {
		sem_wait(sem_children);
	}

	// END

	clean_resources();
	return 0;
}
/*** End of file proj2.c ***/