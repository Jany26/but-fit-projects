/**
 *  ISA Course - Project assignment (Network applications and network administration)
 *  @file argparse.cpp
 *  @author Ján Maťufka (xmatuf00@stud.fit.vutbr.cz)
 * 
 *  @brief Implementation of ArgParse class methods (CLI parsing).
 */

#include "argparse.h"

void ArgParser::print_help(void) {
    using namespace std;
    cerr << "usage: client [ <option> ... ] <command> [<args>] ...\n\n";
    cerr << "<option> is one of\n\n";
    cerr << "  -a <addr>, --address <addr>\n";
    cerr << "     Server hostname or address to connect to\n";
    cerr << "  -p <port>, --port <port>\n";
    cerr << "     Server port to connect to\n";
    cerr << "  -h, --help\n";
    cerr << "     Show this help\n\n";
    cerr << " Supported commands:\n";
    cerr << "   register <username> <password>\n";
    cerr << "   login <username> <password>\n";
    cerr << "   list\n";
    cerr << "   send <recipient> <subject> <body>\n";
    cerr << "   fetch <id>\n";
    cerr << "   logout\n";
}


// expected arguments for commands from command_t enum
const int expected_args[6] = {2, 2, 0, 3, 1, 0};

const std::string valid_commands[6] = {
    "register",
    "login",
    "list",
    "send",
    "fetch",
    "logout",
};
/**
 * @brief Checks the string for valid commands; if valid, returns index to array of commands.
 * If invalid, returns -1.
 */
int command_check(char *str) {
    for (int i = 0; i < 6; ++i) {
        if (strcmp(valid_commands[i].c_str(), str) == 0) {
            return i;
        }
    }
    return -1;
}

void ArgParser::parse(int argc, char** argv) {
    int arg_index = 1;
    for (;;) {
        if (arg_index == argc) {
            break;
        }
        /* HELP */
        if (strcmp(argv[arg_index], "--help") == 0 || strcmp(argv[arg_index], "-h") == 0) {
            ArgParser::exit_value = 0;
            print_help();
            return;
        }
        /* PORT */
        else if (strcmp(argv[arg_index], "--port") == 0 || strcmp(argv[arg_index], "-p") == 0) {
            if (arg_index + 1 > argc) {
                std::cerr << "client: the \"-p\" option needs 1 argument, but 0 provided\n";
                ArgParser::exit_value = 1;
                return;
            }
            ArgParser::port = argv[arg_index+1];
            char *pEnd;
            strtol(ArgParser::port.c_str(), &pEnd, 10);
            if (strcmp(pEnd, "") != 0) {
                std::cerr << "ERROR: invalid <port> value\n";
                ArgParser::exit_value = 1;
                return;
            }
            arg_index += 2;
        }
        /* ADDRESS */
        else if (strcmp(argv[arg_index], "--address") == 0 || strcmp(argv[arg_index], "-a") == 0) {
            if (arg_index + 1 > argc) {
                std::cerr << "client: the \"-a\" option needs 1 argument, but 0 provided\n";
                ArgParser::exit_value = 1;
                return;
            }
            ArgParser::address = argv[arg_index+1];
            arg_index += 2;
        }
        /* COMMAND */
        else {
            int command_index = command_check(argv[arg_index]);
            if (command_index != -1) {
                ArgParser::command = argv[arg_index];
                arg_index++;
                int j;
                
                for (j = 0; j < expected_args[command_index]; ++j, ++arg_index) {
                    // if (argc = arg_index) {
                    ArgParser::arguments[j] = argv[arg_index];
                }
                if (argc != arg_index) {
                    std::cerr << "ERROR: Incorrect amount of arguments for command: ";
                    std::cerr << valid_commands[command_index] << ", expected = ";
                    std::cerr << expected_args[command_index] << std::endl;
                    ArgParser::exit_value = 1;
                    return;
                }
            } else {
                std::cerr << "unknown command" << std::endl;
                ArgParser::exit_value = 1;
            }
            return;
        }
    }
    if (ArgParser::command == "") {
        std::cerr << "ERROR: missing command\n";
        ArgParser::exit_value = 1;
    }
    return;
}

/*** End of file argparse.cpp ***/
