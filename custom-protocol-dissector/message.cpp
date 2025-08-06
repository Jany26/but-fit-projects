/**
 *  ISA Course - Project assignment (Network applications and network administration)
 *  @file message.cpp
 *  @author Jan Matufka <xmatuf00@stud.fit.vutbr.cz>
 * 
 *  @brief Helper functions for handling server responses
 *  and for creating proper ISA Protocol client requests.
 */

#include "argparse.h"
#include "message.h"

/** 
 *  Taken from Stack Overflow.
 *  @see "message.h" for source details.
 */
std::string encode_to_base64(const std::string &in) {

    std::string out;

    int val = 0, valb = -6;
    for (unsigned char c : in) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(val>>valb)&0x3F]);
            valb -= 6;
        }
    }
    if (valb>-6) out.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[((val<<8)>>(valb+8))&0x3F]);
    while (out.size()%4) out.push_back('=');
    return out;
}

void create_token_file(std::string response) {
    std::string temp(response);
    std::string token = temp.substr(21, temp.length());
    std::size_t end = token.find(")");
    token = token.substr(0, end);
    std::ofstream tokenfile;
    tokenfile.open("login-token");
    tokenfile << token;
    return;
}

std::string get_user_token() {
    using namespace std;
    ifstream token_file("login-token");
    string user_token;
    user_token.assign(
        std::istreambuf_iterator<char>(token_file),
        std::istreambuf_iterator<char>()
    );
    return user_token;
}

std::string escape_string(std::string arg) {
    std::string result = "";
    for (auto i: arg) {
        switch(i) {
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\t': result += "\\t"; break;
            case '"': result += "\\\""; break;
            default: result += i; break;
        }
    }
    return result;
}

std::string create_request(std::string command, std::string arguments[]) {
    command_t command_id = (command_t) command_check((char *) command.c_str());

    std::string request = "(" + command + " ";

    std::string fields[3] = {
        escape_string(arguments[0]), 
        escape_string(arguments[1]), 
        escape_string(arguments[2])
    };

    switch (command_id) {
    case C_REGISTER: case C_LOGIN:
        request += "\"" + fields[0] + "\" \"" + encode_to_base64(fields[1]) + "\"";
    break;
    case C_SEND:
        /* TODO: get user token from username */
        request += get_user_token() + " \"" + fields[0];
        request += "\" \"" + fields[1] + "\" \"" + fields[2] + "\"";
    break;
    case C_FETCH:
        request += get_user_token() + " " + fields[0];
    break;
    case C_LIST: default: /*case C_LOGOUT*/
        request += get_user_token();
    break;
    }
    request += ")";
    // std::cout << request << std::endl;
    return request;
}

char escape_char(char a) {
    switch(a) {
        case 'n': return '\n';
        case 't': return '\t';
        case '\\': return '\\';
        case '"': return '"';
        case '0': return '\0';
        default: break;
    }
    return ' '; // default
}

std::vector<std::string> get_fields_from_response(std::string msg) {
    std::vector<std::string> result;
    // loading ok/err - from index 1 (skip '(') until index of first ' '
    result.push_back(msg.substr(1, msg.find(' ') - 1));
    std::string current_word;
    int len = msg.length();
    int i = 0;
    bool in_string = false;
    // parsing data fields in "double-quotes"
    while (i < len) {
        if (msg[i] == '"') {
            if (in_string == true) {
                result.push_back(current_word);
                current_word = "";
                in_string = false;    
            } else {
                in_string = true;
            }
        } else if (in_string) {
            current_word += msg[i] == '\\' ? escape_char(msg[i+1]) : msg[i];
            i += msg[i] == '\\' ? 1 : 0;
        }
        i++;
    }
    
    return result;
}

int display_response(std::string command, std::string response) {
    std::vector<std::string> data = get_fields_from_response(response);

    if (data[0] == "ok") {
        if (command == "list") {
            std::cout << "SUCCESS:" << std::endl;
            for (unsigned i = 1; i < data.size(); i+=2) {
                std::cout << i << ":" << std::endl;
                std::cout << "  From: " << data[i] <<std::endl;
                std::cout << "  Subject: " << data[i+1] << std::endl;
            }
        } else if (command == "fetch") {
            std::cout << "SUCCESS:" << std::endl << std::endl;
            std::cout << "From: " << data[1] << std::endl;
            std::cout << "Subject: " << data[2] << std::endl << std::endl;
            std::cout << data[3];
        } else {
            std::cout << "SUCCESS: " << data[1] << std::endl;
            if (command == "login") {
                create_token_file(response);
            }
        }
    } else /* (data[0] == "err") */ {
        std::cout << "ERROR: " << data[1] << std::endl;
    }
    if (data[0] == "ok") return 0;
    else return 1;
}

/*** End of file message.cpp ***/
