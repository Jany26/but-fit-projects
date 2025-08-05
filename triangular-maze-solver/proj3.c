/************************************************************************/
/*                                                                      */
/*                  IZP - Project 3 - "Praca s datovymi strukturami"    */
/*                  Jan Matufka - xmatuf00                              */
/*                  Last change - 2019-12-12 (ver. 8)                   */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

enum directions {LEFT, UP, RIGHT, DOWN};
enum borders {INVALID, L_WALL, R_WALL, H_WALL = 4};

typedef struct {
    int rows;
    int cols;
    unsigned char *cells;
} Map;

void print_help(void) 
{
	printf("--help \n\t= prints out these instructions\n\n");
	printf("--file [maze_file.txt] \n\t= checks the maze for inconsistencies, prints Valid if the data are consistent\n\n");
	printf("--rpath [R] [C] [maze_file.txt] \n\t= traverses the maze using the right hand method\n\n");
	printf("--lpath [R] [C] [maze_file.txt] \n\t= traverses the maze using the left hand method\n\n");
	printf("[maze_file.txt] = text file with the numbers representing the maze\n");
	printf("[R] [C] = coordinates of the entry point, if invalid, the program will return error message\n");
}

// checks for a border in a specified cell defined by the r and c arguments
// returns true, if the cell has a border on that position, returns false if not
bool isborder(Map *map, int r, int c, int border)
{	// if there is a wall, bool gets value 1, 2 or 4, which means true
	bool wall = ((map->cells[c - 1 + ((r - 1) * map->cols)] - '0') & border);
	return wall; 
}

// loads data into map structure + checks for inconsistencies in the map .txt file
// returns true if map is allocated properly and is valid
// returns false if invalid + frees the cell array if the map can not be allocated properly
bool map_load_and_check(Map *map, FILE *f)
{
	//#0A - loading dimensions and allocating memory for cell array
	fscanf(f, "%d %d", &map->rows, &map->cols); // loading map dimensions
	map->cells = malloc((map->rows * map->cols + 1) * sizeof(unsigned char));
	if (map->cells == NULL) {
		free(map->cells);
		return false;
	}
	memset(map->cells, '\0', map->rows * map->cols + 1); // initializing cell array
	//#0B - loading cell array data 
	char c;
	int non_whitespaces = 0;
	while((c = fgetc(f)) != EOF) {
		if (c > ' ') {
			if (non_whitespaces < map->rows * map->cols)
				map->cells[non_whitespaces] = c;
			non_whitespaces++;
		}
	}
	// #1 - checking consistency of map dimensions and the amount of maze cells
	if (non_whitespaces != map->rows * map->cols) return false; // strlen can be used only on char *
	// #2 - checking for invalid characters
	for (unsigned int i = 0; i < strlen((char *)map->cells); i++)
		if (map->cells[i] < '0' || map->cells[i] > '7') return false;
	// #3 - checking cell border consistency
	bool ok = true; // ok remains true while cell borders are consistent
	int row;
	int col;
	// #3A - checking diagonal (vertical) bordere consistency
	for (row = 1; row <= map->rows && ok; row++)
		for (col = 1; col < map->cols && ok; col++)
			ok = (isborder(map, row, col, R_WALL) == isborder(map, row, col + 1, L_WALL));
	// #3B - checking horizontal border consistency
	for (row = 1; row < map->rows && ok; row++)
		for (col = 1; col <= map->cols && ok; col++) 
			if ((row + col) & 1) 
				ok = (isborder(map, row, col, H_WALL) == isborder(map, row + 1, col, H_WALL));	
	return ok;
}

// returns a number representing the wall that the --rpath or --lpath starts with 
// according to the starting position (possible return values = 1, 2, 4 = L_WALL, R_WALL, H_WALL)
// returns 0 (INVALID) if the starting position is invalid (inside the maze or obstructed by walls)
int start_border(Map *map, int r, int c, int leftright) // leftright = 0 for lpath, != 0 for rpath
{
	if (r < 1 || r > map->rows || c < 1 || c > map->cols) return INVALID;
	// return ? border for --rpath : border for --lpath, since RIGHT = true and LEFT = false
	if      (c == 1         && !((r + c) & 1) && !isborder(map, r, c, L_WALL)) return leftright ? R_WALL : H_WALL;
	else if (c == 1         && ((r + c) & 1)  && !isborder(map, r, c, L_WALL)) return leftright ? H_WALL : R_WALL;
	else if (c == map->cols && !((r + c) & 1) && !isborder(map, r, c, R_WALL)) return leftright ? H_WALL : L_WALL;
	else if (c == map->cols && ((r + c) & 1)  && !isborder(map, r, c, R_WALL)) return leftright ? L_WALL : H_WALL;
	else if (r == 1         && !((r + c) & 1) && !isborder(map, r, c, H_WALL)) return leftright ? L_WALL : R_WALL;
	else if (r == map->rows && ((r + c) & 1)  && !isborder(map, r, c, H_WALL)) return leftright ? R_WALL : L_WALL;
	else return INVALID;
}

// this function tries to move in a given direction
// if the position is moved successfuly in the specified direction, the function returns true
bool try_moving(int direction, Map *map, int *row, int *col)
{
	if (direction == LEFT  && !isborder(map, *row, *col, L_WALL)) {
		*col -= 1; return true;
	}
	else if (direction == RIGHT && !isborder(map, *row, *col, R_WALL)) {
		*col += 1; return true;
	}
	else if (direction == UP && !((*row + *col) & 1) && !isborder(map, *row, *col, H_WALL)) {
		*row -= 1; return true;
	}
	else if (direction == DOWN && ((*row + *col) & 1) && !isborder(map, *row, *col, H_WALL)) {
		*row += 1; return true;
	}
	else return false;
}

// main algorithm for maze solving using the right-hand or left-hand method defined by the method argument
// side effect - moves current position by one cell = changing *r or *c value based on the movement
// returns integer representing the border that the algorithm "touches" after making a move
int step(Map *m, int *r, int *c, int touching_border, int method) 
{	// rpath = true (RIGHT = 2), lpath = false (LEFT = 0)
	if (!((*r + *c) & 1)) { // if the cell has the shape of the letter V, "even cell"
		if (touching_border == L_WALL) {
			// method ? --rpath : --lpath
			if (try_moving(LEFT, m, r, c))                return method ? L_WALL : H_WALL;
			if (try_moving(method ? RIGHT : UP, m, r, c)) return method ? H_WALL : L_WALL;
			if (try_moving(method ? UP : RIGHT, m, r, c)) return R_WALL;
		}
		else if (touching_border == R_WALL) {
			if (try_moving(RIGHT, m, r, c))              return method ? H_WALL : R_WALL;
			if (try_moving(method ? UP : LEFT, m, r, c)) return method ? R_WALL : H_WALL;
			if (try_moving(method ? LEFT : UP, m, r, c)) return L_WALL;
		}
		else { // touching_border == H_WALL
			if (try_moving(UP, m, r, c))                    return method ? R_WALL : L_WALL;
			if (try_moving(method ? LEFT : RIGHT, m, r, c)) return method ? L_WALL : R_WALL;
			if (try_moving(method ? RIGHT : LEFT, m, r, c)) return H_WALL;
		}
	}
	else { // the cell has the shape of the letter A, "uneven cell"
		if (touching_border == L_WALL) {
			if (try_moving(LEFT, m, r, c))                  return method ? H_WALL : L_WALL;
			if (try_moving(method ? DOWN : RIGHT, m, r, c)) return method ? L_WALL : H_WALL;
			if (try_moving(method ? RIGHT : DOWN, m, r, c)) return R_WALL;
		}
		else if (touching_border == R_WALL) {
			if (try_moving(RIGHT, m, r, c))                return method ? R_WALL : H_WALL;
			if (try_moving(method ? LEFT : DOWN, m, r, c)) return method ? H_WALL : R_WALL;
			if (try_moving(method ? DOWN : LEFT, m, r, c)) return L_WALL;
		}
		else { // touching_border == H_WALL
			if (try_moving(DOWN, m, r, c))                  return method ? L_WALL : R_WALL;
			if (try_moving(method ? RIGHT : LEFT, m, r, c)) return method ? R_WALL : L_WALL;
			if (try_moving(method ? LEFT : RIGHT, m, r, c)) return H_WALL;
		}
	}
	return INVALID;
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "error: no arguments\n");
		return 1;
	}
	// --help
	if (!strcmp(argv[1], "--help")) {
		print_help();
		return 0;
	}
	
	// checking arguments content
	bool test = !strcmp(argv[1], "--test");
	bool rpath = !strcmp(argv[1], "--rpath");
	bool lpath = !strcmp(argv[1], "--lpath");	
	if (!test && !rpath && !lpath) {
		fprintf(stderr, "error: invalid argument\n");
		return 1;
	}

	// checking argument count
	if (test && argc < 3) {
		fprintf(stderr, "error: file name missing\n");
		return 1;
	}
	if ((rpath || lpath) && argc < 5) {
		fprintf(stderr, "error: not enough arguments\n");
		return 1;
	}

	// opening file
	FILE *f;
	if (test) f = fopen(argv[2], "r");
	else f = fopen(argv[4], "r");
	if (f == NULL) {
		fprintf(stderr, "error: could not open file\n"); 
		return 1;
	}

	// loading and checking the map
	Map maze;
	bool map_is_valid = (map_load_and_check(&maze, f));
	fclose(f);

	// --test
	if (test) {
		if (map_is_valid) {
			printf("Valid.\n"); 
			free(maze.cells);
			return 0;
		}
		else {
			printf("Invalid.\n");
			free(maze.cells);
			return 1;
		}
	}
	// can not use the right hand or left hand method, if the map is invalid
	if (!map_is_valid) {
		fprintf(stderr, "error: can not go thorugh an invalid map\n");
		free(maze.cells);
		return 1;
	}
	// --lpath or --rpath
	int row_pos = atoi(argv[2]); 
	int col_pos = atoi(argv[3]); // current coordinates
	int border_position = start_border(&maze, row_pos, col_pos, rpath ? RIGHT : LEFT);
	
	if (border_position == INVALID) {
		fprintf(stderr, "error: invalid starting position\n");
		free(maze.cells);
		return 1;
	}
	bool out = false;
	while (!out) // main cycle
	{
		printf("%d,%d\n", row_pos, col_pos); // output
		// take a step (based on border) and set a new border
		border_position = step(&maze, &row_pos, &col_pos, border_position, rpath ? RIGHT : LEFT);
		if (border_position == INVALID) { // checking for possible error
			fprintf(stderr, "error: detected invalid border");
			free(maze.cells);
			return 1;
		}
		// is the position out of the maze ?
		out = (row_pos < 1 || row_pos > maze.rows || col_pos < 1 || col_pos > maze.cols);
	}
	free(maze.cells);
	return 0;
}
