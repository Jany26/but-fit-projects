/**
 *  ISA Course - Project assignment (Network applications and network administration)
 *  @file argparse.h
 *  @author Ján Maťufka (xmatuf00@stud.fit.vutbr.cz)
 *  
 *  @brief Structures and function prototypes used in parsing CLI arguments.
 */
#ifndef __ARGPARSE_H__
#define __ARGPARSE_H__

#include <string.h>
#include <iostream>
#include <string>
#include <fstream>  
#include <vector>

/**
 *  @brief Describes indexes for each valid command.
 *  Also used in "message.h" module (function  create_request()).
 */
typedef enum command {
    C_REGISTER, 
    C_LOGIN, 
    C_LIST, 
    C_SEND, 
    C_FETCH, 
    C_LOGOUT
} command_t;

/**
 *  @brief Expected number of arguments for each
 */
extern const int expected_args[6];

/**
 *  @brief Valid command strings. 
 * 
 *  Index i for command corresponds to index in expected_args array.
 *  If command 'register' is at index 0 here, integer in expected_args
 *  at index 0 will contain expected amount of arguments for that command.
 */
extern const std::string valid_commands[6];

int command_check(char *str);

/**
 *  @brief Contains data obtained by parsing CLI arguments.
 */
class ArgParser {
public:
    int exit_value = 0;
    std::string port = "32323";
    std::string address = "localhost";
    std::string command = "";
    std::string arguments[3] = {"", "", ""};

    /**
     *  @brief Performs the CLI parsing and loading the data into class variables.
     */
    void parse(int argc, char **argv);

    /**
     *  @brief Prints out basic info about client usage to stderr.
     */
    void print_help(void);
};

#endif // __ARGPARSE_H__

/*** End of file argparse.h ***/
