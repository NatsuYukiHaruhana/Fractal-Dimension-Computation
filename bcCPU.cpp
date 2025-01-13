//===============================================================================
// This file is part of the software Fast Box-Counting:
// https://www.ugr.es/~demiras/fbc
//
// Copyright(c)2022 University of Granada - SPAIN
//
// FOR RESEARCH PURPOSES ONLY.THE SOFTWARE IS PROVIDED "AS IS," AND THE
// UNIVERSITY OF GRANADA DO NOT MAKE ANY WARRANTY, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE, NOR DO THEY ASSUME ANY LIABILITY OR RESPONSIBILITY FOR THE USE
// OF THIS SOFTWARE.
//
// For more information see license.html
// ===============================================================================
//
// Authors: Juan Ruiz de Miras and Miguel Ángel Posadas, 2022
// Contact: demiras@ugr.es

#include "bcCPU.h"

void seqBC2D(unsigned char* M, const int m, float* n) {
	unsigned int s = 2;
	unsigned int size = m;
	unsigned char ni = 0; 

	while (size > 2) {
		int sm = s >> 1; // s/2
		unsigned long im;
		unsigned long ismm;

		n[ni] = 0;
		for (unsigned long i = 0; i < (m - 1); i += s) {
			im = i * m;
			ismm = (i + sm) * m;
			for (unsigned long j = 0; j < (m - 1); j += s) {
				M[im + j] = M[(im)+j] || M[(im)+(j + sm)] || M[ismm + j] || M[ismm + (j + sm)];
				n[ni] += M[im + j];
			}
		}
		ni++;
		s <<= 1; // s *= 2;
		size >>= 1; // size /= 2;
	}
}
