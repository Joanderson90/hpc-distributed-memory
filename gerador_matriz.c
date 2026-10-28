/*
 * ============================================================================
 * Gerador da matriz de entrada (arquivo binario) para o trabalho MPI
 * ============================================================================
 * Escreve uma matriz N x N de doubles, em ordem row-major, sem cabecalho
 * (N*N*8 bytes), no mesmo formato lido pelo programa serial e pelo MPI.
 *
 * Os elementos sao pseudo-aleatorios em [-1, 1) e deterministicos: A[i][j] e'
 * funcao de (i, j, semente). Nao e' diagonalmente dominante, para que o
 * pivotamento parcial realmente troque linhas.
 *
 * Compilacao:  gcc -O2 -g -o gerador_matriz gerador_matriz.c
 * Uso:         ./gerador_matriz <N> <arquivo.bin> [semente]
 * ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static double gen_elem(uint64_t semente, long i, long j)
{
    uint64_t x = semente ^ ((uint64_t)i * 0x9E3779B97F4A7C15ULL)
                         ^ ((uint64_t)j * 0xC2B2AE3D27D4EB4FULL);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return ((double)(x >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Uso: %s <N> <arquivo.bin> [semente]\n", argv[0]);
        return 1;
    }
    long N = atol(argv[1]);
    const char *caminho = argv[2];
    uint64_t semente = (argc > 3) ? strtoull(argv[3], NULL, 10) : 42ULL;
    if (N <= 0)
    {
        fprintf(stderr, "Erro: N deve ser positivo\n");
        return 1;
    }

    FILE *f = fopen(caminho, "wb");
    if (!f)
    {
        fprintf(stderr, "Erro: nao foi possivel criar '%s'\n", caminho);
        return 1;
    }
    double *linha = (double *)malloc((size_t)N * sizeof(double));
    if (!linha)
    {
        fclose(f);
        return 1;
    }

    time_t t0 = time(NULL);
    for (long i = 0; i < N; i++)
    {
        for (long j = 0; j < N; j++)
            linha[j] = gen_elem(semente, i, j);
        if (fwrite(linha, sizeof(double), (size_t)N, f) != (size_t)N)
        {
            fprintf(stderr, "Erro de escrita na linha %ld\n", i);
            free(linha);
            fclose(f);
            return 1;
        }
    }
    free(linha);
    fclose(f);
    printf("Matriz %ldx%ld (semente %llu) gravada em '%s' (%.1f MB) em %ld s\n", N, N,
           (unsigned long long)semente, caminho, (double)N * N * 8 / 1e6, (long)(time(NULL) - t0));
    return 0;
}
