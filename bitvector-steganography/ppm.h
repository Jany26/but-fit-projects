//////////////////////////////////////////////////////////////
//  file:           ppm.h                                   //
//  purpose:        IJC-DU1, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-03-18                              //
//////////////////////////////////////////////////////////////

#include "error.h"

#ifndef PPM_H
#define PPM_H

#define IMPLEMENTATION_SIZE_LIMIT (8000 * 8000 * 3)

struct ppm {
        unsigned xsize;
        unsigned ysize;
        char data[];
};


// allocates ppm structure and loads .ppm format specifics (dimensions, binary data)
// into ppm structure (default specifications = P6 variant, color depth = 255)
// ppm file is defined by "filename" argument
// returns NULL if an error is encountered (wrong format, inconsistent data)
// if the data is loaded successfully, pointer to the ppm structure is returned
struct ppm *ppm_read(const char *filename);

// frees dynamic memory allocated via ppm_read
void ppm_free(struct ppm *p);

#endif