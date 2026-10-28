/*
 * ============================================================================
 * Algoritmo: Inversao de Matriz (Gauss-Jordan) -- Paralelo com MPI
 * ============================================================================
 *
 * Mesma eliminacao do programa_serial.c (pivotamento parcial, troca fisica de
 * linhas), com a matriz aumentada [A | I] dividida em blocos contiguos de
 * linhas, um por rank. A cada iteracao k o pivo global e' escolhido com
 * MPI_Allreduce (MAXLOC), a linha pivo e' distribuida com MPI_Bcast e cada
 * rank atualiza apenas as suas linhas.
 *
 * As operacoes sobre cada elemento sao as do serial, na mesma ordem: a
 * inversa e' identica a da versao serial, qualquer que seja P.
 *
 * Tempos por rank (MPI_Wtime): Tcomm = tempo dentro de chamadas MPI (inclui a
 * espera pelos ranks mais lentos); Tcomp = tempo fora delas; Ttotal = soma.
 *
 * Compilacao:
 *   mpicc -O2 -g -o programa_mpi programa_mpi.c
 *
 * Uso (--comparar reune a inversa no rank 0 e a compara, elemento a elemento,
 * com a gravada pelo serial):
 *   mpirun -np P -machinefile mf ./programa_mpi <N> <matriz.bin> [--comparar inversa_serial.bin]
 * ============================================================================
 */

#define _FILE_OFFSET_BITS 64
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------------- */
/* Divisao das N linhas entre P ranks (blocos contiguos)                   */
/* ---------------------------------------------------------------------- */
typedef struct
{
    int N, P;
    int base, resto;
} Particao;

static int linhas_do_rank(const Particao *pt, int r)
{
    return pt->base + (r < pt->resto ? 1 : 0);
}

static int primeira_linha(const Particao *pt, int r)
{
    return r * pt->base + (r < pt->resto ? r : pt->resto);
}

static int dono_da_linha(const Particao *pt, int g)
{
    int corte = pt->resto * (pt->base + 1);
    if (g < corte)
        return g / (pt->base + 1);
    return pt->resto + (g - corte) / pt->base;
}

/* ---------------------------------------------------------------------- */
/* Cronometros: os trechos entre duas marcas sao atribuidos a Tcomp (fora  */
/* de MPI) ou a Tcomm (dentro de MPI), de modo que Tcomp + Tcomm = Ttotal. */
/* ---------------------------------------------------------------------- */
typedef struct
{
    double comp, comm;
    double allreduce, sendrecv, bcast, reduce;
    long n_allreduce, n_sendrecv, n_bcast;
} Tempos;

static Tempos T;
static double g_marca;

#define FECHA_COMP()                      \
    do                                    \
    {                                     \
        double _t = MPI_Wtime();          \
        T.comp += _t - g_marca;           \
        g_marca = _t;                     \
    } while (0)

#define FECHA_COMM(campo)                 \
    do                                    \
    {                                     \
        double _t = MPI_Wtime();          \
        T.campo += _t - g_marca;          \
        T.comm += _t - g_marca;           \
        g_marca = _t;                     \
    } while (0)

/* raiz quadrada por Newton: evita depender de -lm na compilacao */
static double raiz(double x)
{
    if (x <= 0.0)
        return 0.0;
    double r = x > 1.0 ? x : 1.0;
    for (int i = 0; i < 80; i++)
        r = 0.5 * (r + x / r);
    return r;
}

typedef struct
{
    double max, min, media, desvio;
} Estat;

static Estat estatistica(const double *v, int P)
{
    Estat e = {v[0], v[0], 0.0, 0.0};
    for (int r = 0; r < P; r++)
    {
        if (v[r] > e.max)
            e.max = v[r];
        if (v[r] < e.min)
            e.min = v[r];
        e.media += v[r];
    }
    e.media /= P;
    double var = 0.0;
    for (int r = 0; r < P; r++)
        var += (v[r] - e.media) * (v[r] - e.media);
    e.desvio = raiz(var / P);
    return e;
}

static void abortar(const char *msg)
{
    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    fprintf(stderr, "[rank %d] Erro: %s\n", r, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/* ---------------------------------------------------------------------- */
/* Leitura das linhas [g0, g0+n) do arquivo para a metade esquerda de M    */
/* ---------------------------------------------------------------------- */
static void ler_minhas_linhas(double *M, int N, int g0, int n, const char *caminho)
{
    size_t W = 2 * (size_t)N;
    FILE *f = fopen(caminho, "rb");
    if (!f)
        abortar("nao foi possivel abrir o arquivo da matriz");

    fseeko(f, 0, SEEK_END);
    off_t tam = ftello(f);
    off_t esperado = (off_t)N * N * (off_t)sizeof(double);
    if (tam != esperado)
        abortar("tamanho do arquivo diferente de N*N*8 bytes (N nao confere)");

    fseeko(f, (off_t)g0 * N * (off_t)sizeof(double), SEEK_SET);
    for (int li = 0; li < n; li++)
    {
        if (fread(M + (size_t)li * W, sizeof(double), (size_t)N, f) != (size_t)N)
            abortar("falha de leitura do arquivo da matriz");
    }
    fclose(f);
}

/* Metade direita local recebe a parte da identidade correspondente */
static void montar_identidade(double *M, int N, int g0, int n)
{
    size_t W = 2 * (size_t)N;
    for (int li = 0; li < n; li++)
    {
        double *dir = M + (size_t)li * W + N;
        for (int j = 0; j < N; j++)
            dir[j] = (j == g0 + li) ? 1.0 : 0.0;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 3)
    {
        if (rank == 0)
            fprintf(stderr, "Uso: %s <N> <matriz.bin> [--comparar inversa_serial.bin]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    const char *caminho = argv[2];
    const char *arq_serial = NULL;
    for (int a = 3; a < argc; a++)
        if (strcmp(argv[a], "--comparar") == 0 && a + 1 < argc)
            arq_serial = argv[++a];
    if (N <= 0)
        abortar("N deve ser positivo");

    Particao pt = {N, P, N / P, N % P};
    int n_loc = linhas_do_rank(&pt, rank);
    int g0 = primeira_linha(&pt, rank);
    size_t W = 2 * (size_t)N;

    double *M = (double *)malloc((size_t)(n_loc > 0 ? n_loc : 1) * W * sizeof(double));
    double *linha_pivo = (double *)malloc(W * sizeof(double));
    double *tmp = (double *)malloc(W * sizeof(double));
    if (!M || !linha_pivo || !tmp)
        abortar("sem memoria para a matriz local");

    /* ---- entrada: cada rank le so as suas linhas (fora da secao medida) ---- */
    double tr0 = MPI_Wtime();
    ler_minhas_linhas(M, N, g0, n_loc, caminho);
    montar_identidade(M, N, g0, n_loc);
    double t_leitura = MPI_Wtime() - tr0;

    /* ================= SECAO PARALELA (entre barreiras) ================= */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    g_marca = t0;

    int trocas = 0, trocas_entre_ranks = 0;

    for (int k = 0; k < N; k++)
    {
        /* 1. candidato local ao pivo (linhas com indice global >= k) */
        struct
        {
            double v;
            int g;
        } cand, melhor;
        cand.v = -1.0;
        cand.g = N;
        for (int li = 0; li < n_loc; li++)
        {
            int g = g0 + li;
            if (g < k)
                continue;
            double v = fabs(M[(size_t)li * W + k]);
            if (v > cand.v)
            {
                cand.v = v;
                cand.g = g;
            }
        }

        /* 2. pivo global: maior valor; empate -> menor indice global */
        FECHA_COMP();
        MPI_Allreduce(&cand, &melhor, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        FECHA_COMM(allreduce);
        T.n_allreduce++;
        int p = melhor.g;

        /* 3. troca fisica das linhas k e p */
        int rk = dono_da_linha(&pt, k);
        int rp = dono_da_linha(&pt, p);
        if (p != k)
        {
            trocas++;
            if (rk == rp)
            {
                if (rank == rk)
                {
                    double *a = M + (size_t)(k - g0) * W;
                    double *b = M + (size_t)(p - g0) * W;
                    for (size_t j = 0; j < W; j++)
                    {
                        double t = a[j];
                        a[j] = b[j];
                        b[j] = t;
                    }
                }
            }
            else
            {
                trocas_entre_ranks++;
                if (rank == rk)
                {
                    double *a = M + (size_t)(k - g0) * W;
                    FECHA_COMP();
                    MPI_Sendrecv(a, (int)W, MPI_DOUBLE, rp, 100, tmp, (int)W, MPI_DOUBLE, rp, 100,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    FECHA_COMM(sendrecv);
                    T.n_sendrecv++;
                    memcpy(a, tmp, W * sizeof(double));
                }
                else if (rank == rp)
                {
                    double *b = M + (size_t)(p - g0) * W;
                    FECHA_COMP();
                    MPI_Sendrecv(b, (int)W, MPI_DOUBLE, rk, 100, tmp, (int)W, MPI_DOUBLE, rk, 100,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    FECHA_COMM(sendrecv);
                    T.n_sendrecv++;
                    memcpy(b, tmp, W * sizeof(double));
                }
            }
        }

        /* 4. o dono da linha k a normaliza e a coloca no buffer */
        if (rank == rk)
        {
            double *lk = M + (size_t)(k - g0) * W;
            double pivo = lk[k];
            for (size_t j = 0; j < W; j++)
                lk[j] /= pivo;
            memcpy(linha_pivo, lk, W * sizeof(double));
        }

        /* 5. todos recebem a linha pivo normalizada */
        FECHA_COMP();
        MPI_Bcast(linha_pivo, (int)W, MPI_DOUBLE, rk, MPI_COMM_WORLD);
        FECHA_COMM(bcast);
        T.n_bcast++;

        /* 6. atualizacao local das linhas i != k */
        for (int li = 0; li < n_loc; li++)
        {
            if (g0 + li == k)
                continue;
            double *lin = M + (size_t)li * W;
            double fator = lin[k];
            if (fator == 0.0)
                continue;
            for (size_t j = 0; j < W; j++)
                lin[j] -= fator * linha_pivo[j];
        }
    }

    /* soma de verificacao: soma dos |elementos| da inversa */
    double soma_loc = 0.0, soma = 0.0;
    for (int li = 0; li < n_loc; li++)
    {
        const double *dir = M + (size_t)li * W + N;
        for (int j = 0; j < N; j++)
            soma_loc += fabs(dir[j]);
    }
    FECHA_COMP();
    MPI_Reduce(&soma_loc, &soma, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    FECHA_COMM(reduce);

    double t_total = g_marca - t0; /* fim do trabalho deste rank (= Tcomp + Tcomm) */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_secao = MPI_Wtime() - t0; /* barreira a barreira */
    double t_espera = t_secao - t_total;
    /* ===================== fim da secao paralela ===================== */

    /* ---- relatorio: cada rank imprime os seus valores (em ordem de rank) ---- */
    char host[MPI_MAX_PROCESSOR_NAME];
    int hl;
    MPI_Get_processor_name(host, &hl);

    if (rank == 0)
        printf("=== Versao MPI (Opcao A) N=%d P=%d ===\n", N, P);
    for (int r = 0; r < P; r++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r)
        {
            printf("RANK r=%d P=%d N=%d host=%s linhas=%d t_comp=%.6f t_comm=%.6f t_total=%.6f "
                   "t_espera=%.6f t_allreduce=%.6f t_sendrecv=%.6f t_bcast=%.6f t_reduce=%.6f "
                   "n_allreduce=%ld n_sendrecv=%ld n_bcast=%ld\n",
                   rank, P, N, host, n_loc, T.comp, T.comm, t_total, t_espera, T.allreduce,
                   T.sendrecv, T.bcast, T.reduce, T.n_allreduce, T.n_sendrecv, T.n_bcast);
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* estatisticas entre ranks (fora da secao medida) */
    enum { K = 9 };
    double meu[K] = {T.comp, T.comm, t_total, T.allreduce, T.sendrecv, T.bcast, T.reduce, t_espera, t_leitura};
    double *todos = (rank == 0) ? (double *)malloc((size_t)P * K * sizeof(double)) : NULL;
    MPI_Gather(meu, K, MPI_DOUBLE, todos, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        static const char *nome[K] = {"Tcomp", "Tcomm", "Ttotal", "  Tcomm: Allreduce", "  Tcomm: Sendrecv",
                                      "  Tcomm: Bcast", "  Tcomm: Reduce", "Tespera (barr. final)",
                                      "Leitura (fora da secao)"};
        Estat est[K];
        double *col = (double *)malloc((size_t)P * sizeof(double));
        printf("\n=== Estatisticas entre os %d ranks (segundos) ===\n", P);
        printf("%-24s %12s %12s %12s %12s\n", "Metrica", "Tmax", "Tmin", "Tmedia", "Desvio");
        for (int m = 0; m < K; m++)
        {
            for (int r = 0; r < P; r++)
                col[r] = todos[(size_t)r * K + m];
            est[m] = estatistica(col, P);
            printf("%-24s %12.6f %12.6f %12.6f %12.6f\n", nome[m], est[m].max, est[m].min, est[m].media,
                   est[m].desvio);
        }
        double fcomm = 100.0 * est[1].max / est[2].max;
        double desbal = est[0].max / est[0].media;
        printf("\nFcomm = Tcomm_max / Ttotal_max            = %.2f %%\n", fcomm);
        printf("Desbalanceamento de Tcomp = Tmax / Tmedia = %.4f  (1 = perfeitamente balanceado)\n", desbal);
        printf("T_secao (barreira a barreira, rank 0)     = %.6f s\n", t_secao);
        printf("Trocas de linha: %d (entre ranks: %d) de %d iteracoes\n", trocas, trocas_entre_ranks, N);
        printf("Checksum (soma |Inv|): %.12e\n", soma);
        printf("RESULT mpi P=%d N=%d t_secao=%.6f tmax_total=%.6f tmax_comp=%.6f tmin_comp=%.6f tmedia_comp=%.6f "
               "desvio_comp=%.6f tmax_comm=%.6f tmedia_comm=%.6f fcomm=%.4f trocas=%d trocas_inter=%d "
               "checksum=%.12e\n",
               P, N, t_secao, est[2].max, est[0].max, est[0].min, est[0].media, est[0].desvio, est[1].max,
               est[1].media, fcomm, trocas, trocas_entre_ranks, soma);
        free(col);
        free(todos);
    }

    /* ---- validacao opcional contra a inversa gravada pelo serial ---- */
    if (arq_serial)
    {
        double *loc = (double *)malloc((size_t)(n_loc > 0 ? n_loc : 1) * N * sizeof(double));
        for (int li = 0; li < n_loc; li++)
            memcpy(loc + (size_t)li * N, M + (size_t)li * W + N, (size_t)N * sizeof(double));

        int *cnt = NULL, *desl = NULL;
        double *todas = NULL;
        if (rank == 0)
        {
            cnt = (int *)malloc(P * sizeof(int));
            desl = (int *)malloc(P * sizeof(int));
            for (int r = 0; r < P; r++)
            {
                cnt[r] = linhas_do_rank(&pt, r) * N;
                desl[r] = primeira_linha(&pt, r) * N;
            }
            todas = (double *)malloc((size_t)N * N * sizeof(double));
        }
        MPI_Gatherv(loc, n_loc * N, MPI_DOUBLE, todas, cnt, desl, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            double *ref = (double *)malloc((size_t)N * N * sizeof(double));
            FILE *f = fopen(arq_serial, "rb");
            int leitura_ok = 0;
            if (!f)
            {
                printf("VALIDACAO: FALHA - arquivo '%s' nao existe ou nao pode ser aberto\n", arq_serial);
            }
            else
            {
                fseeko(f, 0, SEEK_END);
                off_t tam = ftello(f);
                off_t esp = (off_t)N * N * (off_t)sizeof(double);
                if (tam != esp)
                    printf("VALIDACAO: FALHA - '%s' tem %lld bytes, esperado %lld para N=%d (N nao confere)\n",
                           arq_serial, (long long)tam, (long long)esp, N);
                else
                {
                    rewind(f);
                    leitura_ok = (fread(ref, sizeof(double), (size_t)N * N, f) == (size_t)N * N);
                    if (!leitura_ok)
                        printf("VALIDACAO: FALHA - erro de leitura em '%s'\n", arq_serial);
                }
            }
            if (leitura_ok)
            {
                double dmax = 0.0;
                size_t iguais = 0;
                for (size_t e = 0; e < (size_t)N * N; e++)
                {
                    double d = fabs(todas[e] - ref[e]);
                    if (d > dmax)
                        dmax = d;
                    if (todas[e] == ref[e])
                        iguais++;
                }
                printf("VALIDACAO vs serial: max|diff| = %.3e ; elementos identicos: %zu de %zu -> %s\n",
                       dmax, iguais, (size_t)N * N, dmax <= 1e-9 ? "OK" : "FALHOU");
            }
            if (f)
                fclose(f);
            free(ref);
            free(todas);
            free(cnt);
            free(desl);
        }
        free(loc);
    }

    free(M);
    free(linha_pivo);
    free(tmp);
    MPI_Finalize();
    return 0;
}
