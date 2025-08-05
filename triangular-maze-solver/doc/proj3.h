/**
  * @file   proj3.h
  * @author Jan Matufka (xmatuf00)
  * @date   2019-12-22
  */

/**
  * @mainpage Documentation - Project 3 IZP
  *
  * @section intro Introduction
  * 
  * This is the documentation of the third project in the subject IZP (Programming basics) in the first term of bachelor study on FIT BUT.
  * 
  * Author: Jan Matufka (login: xmatuf00)
  * 
  * Date:   December 22nd, 2019
  *
  * Check the files section for more information (under proj3.h).
  */


#ifndef PROJ3_H
#define PROJ3_H

/** 
  * @details The Map struct describes the map of the maze. The char array *cells contains 
  *          3-bit values representing the existence of given borders in the cell.
  *
  * @details Example of the cell defined by the character: Value in cell array - 6. 6 in binary = 110 --> 
  *          The cell has a BRIGHT and one of the following: BBOTTOM or BTOP (specified by the hasbottom() function).
  */
typedef struct {
    int rows;              /**< number of rows of the cells in the maze*/
    int cols;              /**< number of columns of the cells in the maze*/
    unsigned char *cells;  /**< 1 dimensional array representing the cell values of the maze - 
                                the cells contain 3bit values (0 - 7), each bit represents the existence (1) 
                                or the absence (0) of that particular border.
                                The length of the cell array for valid Map structure should be 
                                (r * c) + 1 (extra space for the '\0' character).*/
} Map;


/**
  * @brief   Defines border values by their bitweight.
  *
  * @details Enum values in enum borders represent bit weight of the walls in the cell, 
  *	         which are used to store border data in cells array in Map structure.
  */
enum borders { 
	BLEFT=0x1,  /**< Left border, defined by the weight of the first bit of the 3-bit number (1). 
	                 00X - X being the bit defined by BLEFT. */

	BRIGHT=0x2, /**< Right border, defined by the weight of the second bit of the 3-bit number (2). 
                     0X0 - X being the bit defined by BRIGHT. */

	BTOP=0x4,   /**< Top border, defined by the weight of the third bit of the 3-bit number (4). 
	                 X00 - X being the bit defined by BTOP. 
	                 Not present, if the sum of the cell coordinates is not even. See hasbottom() function*/

	BBOTTOM=0x4 /**< Bottom border, defined by the weight of the third bit of the 3-bit number (4). 
	                 X00 - X being the bit defined by BBOTTOM. 
	                 Not present, if the sum of the cell coordinates is even. See hasbottom() function*/
};


/**
  * @brief   Frees the allocated memory in Map structure.
  *
  * @param   map: pointer to a Map structure, in which the cell array to be freed is located
  *
  * @pre     allocated memory in Map structure for cells char array
  * @post    the cells char array is freed from the heap
  *
  * @return  This function does not return any value.
  */
void free_map(Map *map);


/** 
  * @brief   Loads the Map structure from the file.
  *
  * @param   map: pointer to a Map structure to load data into
  * @param   filename: char array representing the filename of a file to load data from
  *
  * @pre     declared Map structure, existing file with the filename
  * @post    Map is loaded properly, file is closed properly
  *
  * @return  0: allocated and loaded Map correctly, file opened correctly
  * @return  1: error while opening file
  * @return  2: unable to allocate memory for map structure
  * @return  3: parameter error
  */
int load_map(const char *filename, Map *map);


/** 
  * @brief   Function checks if the particular cell has a certain border.
  *
  * @param   map: pointer to a Map structure
  * @param   r: row position of a cell (coordinate)
  * @param   c: column position of a cell (coordinate)
  * @param   border: bit weight of a border which we want to chcek (see borders enum)
  *
  * @pre     loaded Map structure
  * @post    the data in map is not changed
  *
  * @return  true: the border exists in the specified cell
  * @return  false: the border is not in the specified cell, or invalid parameters
  */
bool isborder(Map *map, int r, int c, int border);


/** 
  * @brief   Checks if the cell has a bottom border.
  *
  * @details Because the map is approximated in the triangular field, the cell can have only three
  *          borders, left, right and either top or bottom border. The location of this horizontal border
  *          is determined by the cell position (cell with an even sum of the coordinates have top border).
  *
  * @param   r: row position of a cell (coordinate)
  * @param   c: column position of a cell (coordinate)
  *
  * @pre     loaded Map structure, valid parameters
  * @post    the data in map is not changed
  *
  * @return  true: the cell has a bottom border (BBOTTOM, as specified in the borders enum)
  * @return  false: the cell does not have a bottom border (BBOTTOM), or the cell position is invalid
  */
bool hasbottom(int r, int c);


/**
  * @brief   Sets a first border in starting position.
  *
  *	@details Finds the first border to check based on the starting position and method.
  *          Essential for starting the maze solving algorithm.
  *
  * @param   map: pointer to a Map structure
  * @param   r: starting row position
  * @param   c: starting column postion
  * @param   leftright: integer specifies which method is used (0 - left, 1 - right ???)
  *
  * @pre     loaded Map structure, valid arguments
  * @post    the border to touch is stored in the return value, the data in map is not changed
  * 
  * @return  0: invalid starting position, either the cell is blocked, or is in the middle of the maze
  * @return  BLEFT = 1
  * @return  BRIGHT = 2
  * @return  BTOP or BBOTTOM (based on the result of hasbottom() function) = 4
  */
int start_border(Map *map, int r, int c, int leftright);


/**
  * @brief   Checks the map structure for inconsistencies.
  *
  * @details Validates the map. Checks for inconsistencies between maze dimensions
  *          and the cell array length, invalid characters (everything other than 
  *          whitespaces and 0-7) and inconsistent borders (e.g. if a cell has BRIGHT
  *          and the one to the left does not have BLEFT, it counts as inconsistent data).
  *
  * @param   map: pointer to a Map structure
  *
  * @pre     map is correctly allocated
  * @post    the data in map is not changed
  *
  * @return  0: if the map is valid (no inconsistencies found)
  * @return  1: map is invalid, inconsistency between dimensions and cell array length
  * @return  2: map is invalid, contains invalid characters (other than 0-7 and whitespaces)
  * @return  3: map is invalid, contains inconsistent border data
  */
int check_map(Map *map);


/**
  * @brief   Loads the map from file and checks for inconsistencies.
  *
  * @details For description of the inconsistencies, see the check_map() function.
  *
  * @param   map: pointer to a Map structure
  * @param   filename: pointer to char supposed to represent the name of the file with the map data
  *
  * @pre     existing file with the "filename", declared map structure
  * @post    the map structure is loaded with the values from the .txt file, the file with the "filename" is closed
  *
  * @return  0: map is correctly loaded, file correctly opened, and no inconsistencies found 
  * @return  1: file could not be opened
  * @return  2: cell array in map could not be allocated
  * @return  3: map data is inconsistent
  */
int load_and_check_map(const char *filename, Map *map);


/**
  * @brief   Checks if the current position (specified by r, c) is out of the maze.
  *
  * @param   r: current row position, c: current column position
  * @param   map: pointer to a Map structure
  * @param   r: current row coordinate 
  * @param   c: current column coordinate
  *
  * @pre     map is correctly allocated; r, c are valid positions
  * @post    if return value is true, the maze solving algorithm can end
  *
  * @return  true: the current position is out of the maze boundaries
  * @return  false: we are still located inside the maze (the maze solving has to continue)
  */
bool is_out(Map *map, int r, int c);


/**
  * @brief   Solves the maze and prints the coordinates it moves through.
  *
  * @details Takes starting r, c coordinates and based on the method changes the r and c
  *          accordingly to the path it takes to get out of the maze. After each change of
  *          the cell position prints the position coordinates into stdout.
  *
  * @param   map: pointer to a map structure representing the maze info
  * @param   r: starting row position
  * @param   c: starting column position
  * @param   leftright: defines which method to use to solve the maze (0 for left, non-zero for right)
  *
  * @pre     map is valid, r and c define valid starting coordinates, leftright argument is valid
  * @post    the path out of the maze is printed, no data in the map structure are changed
  *
  * @return  This function does not return any value.
  */
void print_path(Map *map, int r, int c, int leftright);

#endif