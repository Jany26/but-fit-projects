#!/usr/bin/env python3.8
"""
FILE:       fileget.py
TITLE:      IPK Project 1
STUDENT:    Ján Maťufka <xmatuf00@stud.fit.vutbr.cz>
TESTED ON:  Python 3.6 (merlin), Python 3.8.5 (local)
USAGE:      ./fileget.py -n NAMESERVER -f SURL
"""

import socket
import sys
import argparse
import re
import os
from urllib.parse import urlparse

TIMEOUT_SECONDS = 30
BUFFERSIZE_BYTES = 1024
RECURSIVE = False


def parseArguments():
    """ 
        Command line argument parsing. Also checks for some errors.
        Returns 4 values = addressHost, addressName, filePath, serverName
    """
    parser = argparse.ArgumentParser(description = 'Fileget client (IPK - project 1).')
    parser.add_argument("-n", metavar = 'NAMESERVER', type = str, required = True, help = 'IP address and port number for the name server')
    parser.add_argument("-f", metavar = 'SURL', type = str, required = True, help = 'SURL of the file to be downloaded (expecting FSP protocol)')
    args = parser.parse_args()

    data = args.n.split(":")
    addressHost, addressPort = data[0], int(data[1])
    surl = urlparse(args.f) # scheme, netloc, path
    nameServer = str(surl.netloc)
    filePath = str(surl.path)

    if not surl.scheme == "fsp":
        sys.exit("ERROR: Unsupported URL protocol")
    if not re.match("^[\-\_\.a-zA-Z0-9]+$", nameServer):
        sys.exit("ERROR: Invalid nameserver name")
    
    # delete '/' from filePath starting with '/'
    filePath = surl.path[surl.path.startswith("/") and len("/"):]

    return addressHost, addressPort, filePath, surl.netloc



def createArrayOfFiles(filePath, serverName, addressHost, addressPort):
    """
        Creates and returns an array of filePaths in order to download respective files from server.
        Contains more than one item only in case of '*' (GETALL).
    """
    files = []
    if (filePath == "*"):
        global RECURSIVE
        RECURSIVE = True
        TCP_MSG = f"GET index FSP/1.0\r\nHostname: {serverName}\r\nAgent: xmatuf00\r\n\r\n"
        getFileTCP("tempfile", TCP_MSG, addressHost, addressPort)
        with open('tempfile') as file:
            for line in file:
                files.append("./" + line.rstrip())
        os.remove("tempfile")
    else:
        files.append(filePath)

    return files



def parseTCPHeader(packet, fileObject):
    """
        Checks for the header in the packet data and removes it (if needed).
        Only needed for the first received packet.
    """
    if re.match(b"^FSP/1.0\s+Success\s+Length\:\s*[0-9]+\s+", packet):
        parts = re.search(br"Length:\s*([0-9]+)", packet)
        fileLength = int(parts.group(1))
        packet = re.sub(b"^FSP/1.0\s+Success\s+Length\:[0-9]+\s+", b"", packet)

    elif re.match(b"^FSP/1.0\s+Bad\s+Request\s+", packet):
        sys.exit("ERROR: Invalid Request (Possible Syntax Error)")

    elif re.match(b"^FSP/1.0\s+Not\s+Found\s+Length\:\s*[0-9]+\s+", packet):
        sys.exit("ERROR: File Not Found")

    elif re.match(b"^FSP/1.0\s+Server\s+Error\s+", packet):
        sys.exit("ERROR: Unspecified Server Error")

    return fileLength, packet



def parseUDPAnswer(packet):
    """
        Checks the response from nameserver. Exit in case of errors.
        Returns File System Address and Port in case of OK response.
    """
    if re.match(b"^OK\s+", packet):
        parts = re.sub(b"^OK\s+", b"", packet).split(b":")
        host, port = parts[0], int(parts[1])

    elif packet == "ERR Not Found":
        sys.exit("ERROR: Server address was not found on the nameserver")

    elif packet == "ERR Syntax":
        sys.exit("ERROR: Wrong request syntax")

    else:
        sys.exit("ERROR: Unexpected answer from the nameserver")

    return host, port



def getDataUDP(message, addressHost, addressPort): 
    """
        Getting the IP address from the nameserver.
        Raising exceptions when the file is not found.

        message         request for data
        addressHost     from command line argument
        addressPort     from command line argument
        return          addressHost, addressPort = tuple containing the IPv4 address of the nameserver
        
        note - UDP works with the nameserver
    """
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        udp.settimeout(TIMEOUT_SECONDS)
        udp.sendto(bytes(message, "utf-8"), (addressHost, addressPort))
        dataUDP = udp.recv(BUFFERSIZE_BYTES)
        host, port = parseUDPAnswer(dataUDP)
        udp.close()
    except socket.timeout:
        udp.close()
        sys.exit("ERROR: Server timeout (UDP)")
    return host, port



def getFileTCP(filePath, message, addressHost, addressPort):
    """
        Get a file from the server using TCP protocol.

        filePath        what is the resulting file going to be called
        message         request for data from server
        addressHost     Address where is the file system
        addressPort     Port where is the file system
        
        note - TCP Works with the file system.
    """
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        tcp.settimeout(TIMEOUT_SECONDS)
        tcp.connect((addressHost, addressPort))
        tcp.send(message.encode("utf-8"))

        dataTCP = tcp.recv(BUFFERSIZE_BYTES)
        # print(dataTCP)
        expectedFileLength, dataTCP = parseTCPHeader(dataTCP, filePath)
        realFileLength = len(dataTCP)
        file = open(filePath, "wb")
        file.write(dataTCP)
        while True:
            dataTCP = tcp.recv(BUFFERSIZE_BYTES)
            # print(dataTCP)
            realFileLength += len(dataTCP)
            file.write(dataTCP)
            if not dataTCP: 
                break
        file.close()
        tcp.close()
        if (realFileLength != expectedFileLength):
            sys.exit("ERROR: Connection unexpectedly ended. File size not consistent with the header")
    except socket.timeout:
        file.close()
        tcp.close()
        sys.exit("ERROR: Server timeout (TCP)")



def main():
    UDPaddressHost, UDPaddressPort, filePath, serverName = parseArguments()
    
    UDP_MSG = f"WHEREIS {serverName}"
    TCPaddressHost, TCPaddressPort = getDataUDP(UDP_MSG, UDPaddressHost, UDPaddressPort)
    
    files = createArrayOfFiles(filePath, serverName, TCPaddressHost, TCPaddressPort)
    for f in files:
        (filePath, fileName) = os.path.split(f)
        if RECURSIVE:
            os.makedirs(filePath, exist_ok = True)
            fileName = f
        TCP_MSG = f"GET {f} FSP/1.0\r\nHostname: {serverName}\r\nAgent: xmatuf00\r\n\r\n"
        getFileTCP(fileName, TCP_MSG, TCPaddressHost, TCPaddressPort)
    print("SUCCESS")
        
if __name__ == '__main__':
    main()
