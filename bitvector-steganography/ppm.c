//////////////////////////////////////////////////////////////
//  file:           ppm.c                                   //
//  purpose:        IJC-DU1, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-03-18                              //
//////////////////////////////////////////////////////////////

#include "ppm.h"

struct ppm *ppm_read(const char *filename)
{
	FILE *file_ptr = fopen(filename, "r");
	if (file_ptr == NULL) {
		warning_msg("fopen(%s): Nepovedlo se nacist soubor", filename);
		goto error_handler_before_fopen;
	}

	// only works with P6 format
	if (fgetc(file_ptr) != 'P' || fgetc(file_ptr) != '6') {
		warning_msg("%s: Chybny format souboru", filename);
		goto error_handler;
	}

	unsigned picture_width, picture_height, color_val;
	
	// loading three numbers while ignoring whitespaces
	if (fscanf(file_ptr, " %u %u %u", &picture_width, &picture_height, &color_val) != 3) {
		warning_msg("%s: Nepovedlo se nacist rozmery souboru", filename);
		goto error_handler;
	}

	if (color_val != 255) {
		warning_msg("Nepodporovany barevny rezim");
		goto error_handler;
	}

	// skipping another whitespace until the pixel RGB data can be loaded
	fgetc(file_ptr);
		
	struct ppm *picture = malloc(sizeof(struct ppm) + (sizeof(char) * 3 * picture_width * picture_height));
	
	if (picture == NULL) {
		warning_msg("malloc: Chyba alokace pameti (%lu bajtu binarnich dat)", 3 * picture_width * picture_height);
		goto error_handler;
	}

	picture->xsize = picture_width;
	picture->ysize = picture_height;
	
	// only works for pictures smaller than 8000 x 8000 pixels as per the implementation limit
	if (3 * picture->xsize * picture->ysize > IMPLEMENTATION_SIZE_LIMIT) {
		warning_msg("Rozmery %u x %u presahli implementacni limit %lu",
			picture->xsize, picture->ysize, (unsigned long) IMPLEMENTATION_SIZE_LIMIT);
		goto error_handler_after_alloc;
	}
	
	// RGB data loading
	int c;
	unsigned long counter = 0;
	while ((c = fgetc(file_ptr)) != EOF) {
		picture->data[counter] = c;
		counter++;
	}

	// checking for data inconsistencies	
	if (counter != 3 * picture->xsize * picture->ysize) {
		warning_msg("Nekonzistenti data - realni velikost neodpovida rozmerum");
		goto error_handler_after_alloc;
	}

	fclose(file_ptr);
	return picture;
	
	error_handler_after_alloc:	
		ppm_free(picture);
	
	error_handler:
		fclose(file_ptr);
	
	error_handler_before_fopen:
		return NULL;
}

void ppm_free(struct ppm *p)
{
	free(p);
}