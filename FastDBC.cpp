#include "FastDBC.h"
#include <algorithm>
#include <cstring>

template <typename T>
T max4(T a, T b, T c, T d) {
    return std::max(std::max(a, b), std::max(c, d));
}

template <typename T>
T min4(T a, T b, T c, T d) {
    return std::min(std::min(a, b), std::min(c, d));
}

void seqDBC(unsigned char* I, const int M, const unsigned char G, float Nr[]) {
    unsigned char *Imax = new unsigned char[M * M];
    unsigned char *Imin = new unsigned char[M * M];

    memcpy(Imax, I, sizeof(unsigned char) * M * M);
    memcpy(Imin, I, sizeof(unsigned char) * M * M);

    unsigned int s = 2; // grid size is s x s
    unsigned int size = M;
    unsigned char Nri = 0; // index for array Nr

    while (size > 2) {
        float h = (G * s) / M; // box height

        h = h == 0.0 ? 0.001 : h;

        for (int i = 0; i < (M - 1); i += s) {
            for (int j = 0; j < (M - 1); j += s) {
                Imax[i * M + j] = max4(Imax[i * M + j], Imax[i * M + j + s / 2], Imax[(i + s / 2) * M + j], Imax[(i + s / 2) * M + j + s / 2]);
                Imin[i * M + j] = min4(Imin[i * M + j], Imin[i * M + j + s / 2], Imin[(i + s / 2) * M + j], Imin[(i + s / 2) * M + j + s / 2]);

                Nr[Nri] = Nr[Nri] + (Imax[i * M + j] / h - Imin[i * M + j] / h + 1); // Nr computed as in Equations 2 and 3
            }
        }

        Nri++;
        s *= 2;
        size /= 2;
    }
}
