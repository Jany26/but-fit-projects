/**
 *  ISA Course - Project assignment (Network applications and network administration)
 *  @file main.cpp
 *  @author Jan Matufka <xmatuf00@stud.fit.vutbr.cz>
 * 
 *  @brief Basic client implementation for ISA Protocol.
 */

#include "message.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>

/**
 *  Creates a TCP socket which can send/recieve data
 *  in the ISA Protocol format.
 * Code snippets for preparing the socket and address info taken from:
 *  (slightly customized to suit project's needs)
 * 
 *  @see https://beej.us/guide/bgnet/html/#client-server-background
 *  -- Beej's Guide to Network Programming
 *  -- author - Brian “Beej Jorgensen” Hall
 *  -- November 20, 2020
 *  -- cited: 2021-11-14
 */
int main(int argc, char **argv) {
    using namespace std;
    ArgParser args;
    args.parse(argc, argv);
    
    if (args.exit_value) {
        return args.exit_value;
    }

    /* start of citation starts here (slight modifications to the code were added) */
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC; // ipv4 and ipv6
    hints.ai_socktype = SOCK_STREAM; // tcp
    struct addrinfo *resolver;
    
    int status;
    if ((status = getaddrinfo(args.address.c_str(), args.port.c_str(), &hints, &resolver)) != 0) {
        cerr << "ERROR: unknown address or port " << gai_strerror(status) << endl;
        return 2;
    }

    int socket_fd = socket(resolver->ai_family, resolver->ai_socktype, resolver->ai_protocol);
    if (socket_fd == -1) {
        cerr << "ERROR: Couldnt open the socket.\n";
        return 1;
    }
    bind(socket_fd, resolver->ai_addr, resolver->ai_addrlen);

    int yes = 1;
    if (setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) == -1) {
        perror("setsockopt");
        return 1;
    }
    /* citation ends here */

    // create string to send to server
    char payload[4192];
    memset(payload, '\0', 4192);

    // preparing request string
    string request = create_request(args.command, args.arguments);
    strcpy(payload, request.c_str());
    
    // prepare response string
    char response[4192];
    memset(response, '\0', 4192);
    
    // preparing request string
    connect(socket_fd, resolver->ai_addr, resolver->ai_addrlen);
    freeaddrinfo(resolver);

    // prepare variables for send() call
    int length = strlen(payload);
    int bytes_received;
    int flags = 0;
    
    send(socket_fd, payload, length, flags);

    std::string full_response = "";
    do {
        bytes_received = recv(socket_fd, response, 4192, flags);
        full_response += response;
        memset(response, '\0', 4192);
    } while (bytes_received != 0);
    
    args.exit_value = display_response(args.command, full_response );
    return args.exit_value;
}

/*** End of main.cpp ***/
