/**
 * @file analyze-elf.c
 * @author Jan Matufka (xmatuf00@stud.fit.vutbr.cz)
 * @brief This program implements one feature from the readelf utility.
 * It displays the list of ELF program headers (segments), 
 * along with each section corresponding to the particular segment.
 * @date 2022-03-27
 */


#include <fcntl.h>
#include <gelf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/**
 * @brief Converts a p_type value (number) 
 * to a string specifying the program header name.
 */
void phdr_getname(size_t p_type, char *result) 
{
    switch(p_type) {
        case PT_NULL: sprintf(result, "NULL"); break;
        case PT_LOAD: sprintf(result, "LOAD"); break;
        case PT_DYNAMIC: sprintf(result, "DYNAMIC"); break;
        case PT_INTERP: sprintf(result, "INTERP"); break;
        case PT_NOTE: sprintf(result, "NOTE"); break;
        case PT_SHLIB: sprintf(result, "SH_LIB"); break;
        case PT_PHDR: sprintf(result, "PHDR"); break;
        default: sprintf(result, "0x%lx", p_type); break;
    }
}

/**
 * @brief Converts a p_flags value (number) 
 * to a string specifying the program header permissions.
 */
void phdr_getpermissions(size_t flags, char *result) 
{
    result[0] = (flags & 0x4) ? 'R' : '-';
    result[1] = (flags & 0x2) ? 'W' : '-';
    result[2] = (flags & 0x1) ? 'X' : '-';
    result[3] = '\0';
}

/**
 * @note Big part of this program was inspired by examples from:
 * "libelf by Example" by Joesph Koshy (2010), source:
 * @see https://atakua.org/old-wp/wp-content/uploads/2015/03/libelf-by-example-20100112.pdf
 * 
 * @note Implementation details (such as header type values etc.) 
 * were taken from ELF specification:
 * @see http://ftp.openwatcom.org/devel/docs/elf-64-gen.pdf
 */
int main(int argc, char **argv)
{
    int fd;
    Elf *e;
    char *name; // *p; // pc[4*sizeof(char)];
    Elf_Scn *scn;
    GElf_Phdr phdr;
    GElf_Shdr shdr;
    size_t shstrndx;
    size_t n;

    /* INITIAL CHECKS */
    if (argc != 2) { 
        fprintf(stderr, "usage: %s file-name\n", argv[0]);
        return 1;
    }
    /* code taken from libelf by example */
    if (elf_version(EV_CURRENT) == EV_NONE) {
        fprintf(stderr, "ELF library initialization failed\n");
        return 1;
    }
    if ((fd = open(argv[1], O_RDONLY, 0)) < 0) {
        fprintf(stderr, "open \"%s\" failed\n", argv[1]);
        return 1;
    }
    /* INITIALIZING THE ELF FILE */
    if ((e = elf_begin(fd, ELF_C_READ, NULL)) == NULL) {
        fprintf(stderr, "elf_begin() failed: %s\n", elf_errmsg(-1));
        return 1;
    }
    if (elf_kind(e) != ELF_K_ELF) {
        fprintf(stderr, "%s is not an ELF object\n", argv[1]);
        return 1;
    }
    /* PROGRAM HEADERS INIT */
    if (elf_getphdrnum(e, &n) != 0) {
        fprintf(stderr, "elf_getphdrnum() failed: %s", elf_errmsg(-1));
        return 1;
    }
    /* STRING TABLE INDEX INIT */
    if (elf_getshdrstrndx(e, &shstrndx) != 0) {
        fprintf(stderr, "elf_getshdrstrndx() failed: %s", elf_errmsg(-1));
        return 1;
    }
    scn = NULL;
    /* end of code taken from libelf by example */

    printf("%-8s %-12s %-4s %s\n", "Segment", "Type", "Perm", "Sections");
    for (size_t i = 0; i < n; i++) {
        /* checking result of gelf_getphdr() - taken from libelf by example */
        if (gelf_getphdr(e, i, &phdr) != &phdr) {
            fprintf(stderr, "getphdr() failed: %s", elf_errmsg(-1));
            return 1;
        }
        
        /* formatting output */
        char phdr_name[50] = {'\0',};
        phdr_getname(phdr.p_type, phdr_name);
        char phdr_perm[4] = {'\0',};
        phdr_getpermissions(phdr.p_flags, phdr_perm);

        printf("%02ld %*.s %-12s %-5s", i, 5, "", phdr_name, phdr_perm);

        /* ELF section to ELF segment mapping */        
        while ((scn = elf_nextscn(e, scn)) != NULL) {
            if (gelf_getshdr(scn, &shdr) != &shdr) {
                fprintf(stderr, "getshdr() failed: %s\n", elf_errmsg(-1));
                return 1;
            }
            if ((name = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL) {
                fprintf(stderr, "elf_strptr() failed: %s\n", elf_errmsg(-1));
                return 1;
            }
            name = elf_strptr(e, shstrndx, shdr.sh_name);

            /* fixing the disappearing .bss section */
            size_t size = (shdr.sh_type == SHT_NOBITS) ? phdr.p_memsz : phdr.p_filesz;
            
            if ((shdr.sh_offset >= phdr.p_offset && 
                shdr.sh_offset < phdr.p_offset + size)) {
                printf("%s ", name);
            }
        }
        printf("\n");
    }
    elf_end(e);
    close(fd);
    return 0;
}

/*** End of file analyze-elf.c ***/
