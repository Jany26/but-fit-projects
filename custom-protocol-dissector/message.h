/**
 *  ISA Course - Project assignment (Network applications and network administration)
 *  @file message.h
 *  @author Jan Matufka <xmatuf00@stud.fit.vutbr.cz>
 * 
 *  @brief Function headers and documentation for helper functions.
 */

#ifndef __MESSAGE_H__
#define __MESSAGE_H__

#include "argparse.h"

/**
 *  @brief Encodes a string into Base64 form.
 *  @param in String with data needed for encoding.
 *  @return Input string encoded in base64.
 * 
 *  @see https://stackoverflow.com/questions/180947/base64-decode-snippet-in-c
 *  -- answer by Manuel Martinez (2016-01-02 at 21:51)
 *  -- edited by Peter Mortensen (2021-01-09 at 16:53)
 *  -- basically full function was taken from here
 * 
 *  @see used in create_request() function
 */
std::string encode_to_base64(const std::string &in);

/**
 *  @brief Escaping special characters in string.
 *  @param arg input string
 *  @return same string, but with escapable characters replaced with escape sequences
 * 
 *  Even though the user escapes characters in the command line,
 *  shell still converts these sequences back into ASCII chars.
 *  However, client request has to contain those escape sequences, 
 *  so it is important to convert them back.
 *  @see used in create_request() function
 */
std::string escape_string(std::string arg);

/**
 *  @brief Loads user ID from login-token file.
 *  @return Base64 token of user, who is currently logged in.
 * 
 *  @see https://stackoverflow.com/questions/2912520/read-file-contents-into-a-string-in-c
 *  -- answer by Maik Beckmann (2010-05-26 at 11:48)
 *  -- edited by Martijn Pieters (2015-07-18 at 22:19)
 *  -- code snippet for reading file into string was taken from here
 * 
 *  @see used in create_request() function
 * 
 *  Called when used command request requires currently loggen in user ID (base64).
 *  Reads login-token file and returns its content.
 */
std::string get_user_token();

/**
 *  @brief Creates a correctly structured string to send to server.
 *  @param command used command string
 *  @param fields command arguments 
 *  @return string to be sent to the server
 */
std::string create_request(std::string command, std::string arguments[]);

/**
 *  @brief Parses the server response.
 *  @param response string containing full server response payload
 *  @return data fields used in display_response()
 *  @see display_response()
 */
std::vector<std::string> get_fields_from_response(std::string response);

/**
 *  @brief Parses response string and creates login-token file.
 *  @param response response from server after successful login
 *  @see display_response() calls create_token_file()
 *  when parsing login command 'ok' response. 
 */
void create_token_file(std::string response);

/**
 *  @brief Displays the server response according to the reference client.
 *  @param command which command is the server responding to
 *  @param response server response after used command
 *  @return 0 if SUCCESS or 1 if ERROR message encountered
 */
int display_response(std::string command, std::string response);

#endif // __MESSAGE_H__

/*** End of file argparse.h ***/
