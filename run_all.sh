#!/usr/bin/env bash
# =============================================================================
# run_all.sh: reprodutibilidade do trabalho
# Inversao de matriz por eliminacao de Gauss-Jordan com memoria distribuida (MPI)
# TEC508 - Computacao de Alto Desempenho (UEFS)
#
# Reproduz, com os mesmos comandos e a mesma nomenclatura de arquivos da campanha
# de medicao: compilacao, matrizes, serial (corretude, calibracao e linha de base),
# validacao, strong scaling, weak scaling (duas definicoes), versao otimizada,
# consolidacao dos dados e verificacao final dos nos.
#
# ONDE EXECUTAR
#   No node1 do cluster UMBU, em ~/mpi (pasta com os .c). O mpirun e' lancado a
#   partir do node1, com 1 processo por no (machinefile "nodeX slots=1").
#
# COMO USAR (nao executa nada sem argumentos)
#   ./run_all.sh ajuda
#   ./run_all.sh compilar matrizes ocupacao serial validar
#   nohup ./run_all.sh tudo > run_all.log 2>&1 &      # campanha completa
#
# ETAPAS (na ordem de "tudo")
#   compilar    gcc/mpicc dos quatro programas
#   matrizes    gerador_matriz com semente 42 para todos os N usados
#   ocupacao    who + uptime + os 4 processos mais ativos de cada no (secao 7 do enunciado)
#   serial      corretude (N=300 --verify), calibracao (N=1000, 2000, 3000), inversas
#               de referencia e linha de base (N=4000, UMA execucao)
#   validar     inversa MPI e otimizada contra a do serial, elemento a elemento
#   forte       strong scaling: N=4000, P=1,2,4,8,13, 3 execucoes por P
#   fraco       weak scaling: N ~ P^(1/3) e N = 500*P (3 execucoes por ponto)
#   otimizada   versao otimizada: N=4000, P=1,2,4,8,13, 3 execucoes por P
#   perfil      /usr/bin/time -v, mpiP e perf stat (somente se as ferramentas existirem)
#   resumo      media, desvio, speedup e eficiencia do strong scaling
#   consolidar  junta todas as saidas em dados_completos.txt
#   limpeza     confere que nao sobrou processo em nenhum no
#
# TEMPO ESTIMADO DA CAMPANHA COMPLETA (nos ociosos): cerca de 1 h 30 min
#   serial ~7 min; forte ~28 min; fraco ~18 min; otimizada ~21 min; validacao ~5 min.
#   Espaco em disco: ~1 GB de matrizes de entrada, mais as inversas escritas pelo serial.
#
# SOBRE O PROFILING (etapa "perfil")
#   No cluster usado, /usr/bin/time, mpiP e perf NAO estao instalados. Esta etapa
#   so executa o que encontrar e, no nosso ambiente, NAO foi executada: o relatorio
#   declara a limitacao e substitui essas medidas pela instrumentacao manual
#   (MPI_Wtime por rank e por chamada, ja embutida no programa_mpi) e pelo tempo
#   externo registrado com `uptime` em res/carga.log (resolucao de 1 s).
#
# VARIAVEIS (valores padrao entre parenteses; sobrescreva no ambiente)
#   DIR_TRAB ($HOME/mpi)   SEED (42)   REPS (3)   N_FORTE (4000)
#   PS_FORTE ("1 2 4 8 13")   CALIB_NS ("1000 2000 3000")   N_VALID ("300 1000")
#   WEAK_CBRT ("1:2000 2:2520 4:3175 13:4703")   WEAK_LIT ("1:500 2:1000 4:2000 13:6500")
#   HOSTS (node1 ... node13)   MPIRUN_OPTS ("")   SKIP_SSH (0)   FORCAR (0)
#   SEM_MACHINEFILE (0): 1 omite o machinefile; so para testar o script em uma maquina
#   O ponto P=8 (N=4000) das duas series de weak scaling reaproveita as
#   execucoes do strong scaling (res/mpi_P8_r*.out), como no relatorio.
# =============================================================================

set -u

DIR_TRAB=${DIR_TRAB:-$HOME/mpi}
SEED=${SEED:-42}
REPS=${REPS:-3}
N_FORTE=${N_FORTE:-4000}
PS_FORTE=${PS_FORTE:-"1 2 4 8 13"}
CALIB_NS=${CALIB_NS:-"1000 2000 3000"}
N_VALID=${N_VALID:-"300 1000"}
WEAK_CBRT=${WEAK_CBRT:-"1:2000 2:2520 4:3175 13:4703"}
WEAK_LIT=${WEAK_LIT:-"1:500 2:1000 4:2000 13:6500"}
MPIRUN_OPTS=${MPIRUN_OPTS:-""}
SKIP_SSH=${SKIP_SSH:-0}
FORCAR=${FORCAR:-0}
HOSTS=${HOSTS:-$(for n in $(seq 1 13); do printf 'node%s ' "$n"; done)}
read -r -a HOSTS_ARR <<< "$HOSTS"

ajuda() { sed -n '2,55p' "$0" | sed 's/^# \{0,1\}//'; }

# sem argumentos: so mostra a ajuda (nenhum efeito colateral)
if [ $# -eq 0 ]; then ajuda; exit 0; fi

cd "$DIR_TRAB" 2> /dev/null || { echo "ERRO: pasta de trabalho '$DIR_TRAB' inexistente (defina DIR_TRAB)" >&2; exit 1; }
mkdir -p res mf

# ---------------------------------------------------------------- utilitarios
msg() { echo; echo "##### $*"; }

# cria mf/mf_P com os P primeiros nos, 1 processo por no
machinefile() {
    local p=$1 i
    if [ "$p" -gt "${#HOSTS_ARR[@]}" ]; then
        echo "ERRO: P=$p excede o numero de nos (${#HOSTS_ARR[@]}); este roteiro usa 1 processo por no" >&2
        return 1
    fi
    : > "mf/mf_$p"
    for ((i = 0; i < p; i++)); do echo "${HOSTS_ARR[$i]} slots=1" >> "mf/mf_$p"; done
}

# argumentos de colocacao dos processos: machinefile (padrao). SEM_MACHINEFILE=1 omite
# o machinefile (so para testar a logica do script em uma unica maquina).
mf_args() { if [ "${SEM_MACHINEFILE:-0}" = 1 ]; then :; else echo "-machinefile mf/mf_$1"; fi; }

# executar PROGRAMA P N "rotulo no carga.log" arquivo_de_saida
# Registra o uptime antes e depois (tempo externo, resolucao de 1 s) e guarda a saida.
executar() {
    local prog=$1 p=$2 n=$3 tag=$4 saida=$5
    machinefile "$p" || return 1
    echo "$tag inicio $(uptime)" >> res/carga.log
    # shellcheck disable=SC2086
    mpirun $MPIRUN_OPTS -np "$p" $(mf_args "$p") "./$prog" "$n" "m$n.bin" > "$saida" 2>&1
    echo "$tag fim $(uptime)" >> res/carga.log
}

# todos os N que precisam de matriz
todos_n() {
    {
        echo $CALIB_NS $N_FORTE $N_VALID
        local c
        for c in $WEAK_CBRT $WEAK_LIT; do echo "${c#*:}"; done
    } | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -n | uniq
}

tserial() { awk -F= '/^tempo_eliminacao_s/ {print $2}' "stats${N_FORTE}.txt" 2> /dev/null; }

# ---------------------------------------------------------------- etapas
etapa_compilar() {
    msg "compilar"
    gcc -O2 -g -o gerador_matriz gerador_matriz.c || return 1
    gcc -O2 -g -o programa_serial programa_serial.c -lm || return 1 # -lm: sqrt da verificacao opcional (--verify)
    mpicc -O2 -g -o programa_mpi programa_mpi.c || return 1
    mpicc -O2 -g -o programa_otimizado programa_otimizado.c || return 1
    echo "compilou: gerador_matriz, programa_serial, programa_mpi, programa_otimizado"
}

etapa_matrizes() {
    msg "matrizes (semente $SEED)"
    local n
    for n in $(todos_n); do
        if [ -s "m$n.bin" ] && [ "$FORCAR" != 1 ]; then echo "m$n.bin ja existe; mantida"; continue; fi
        ./gerador_matriz "$n" "m$n.bin" "$SEED" > /dev/null || return 1
        echo "m$n.bin gerada ($(du -h "m$n.bin" | cut -f1))"
    done
}

etapa_ocupacao() {
    msg "ocupacao dos nos (carga e processos mais ativos)"
    who
    local h
    for h in "${HOSTS_ARR[@]}"; do
        echo "== $h"
        if [ "$SKIP_SSH" = 1 ]; then echo "(SKIP_SSH=1: ignorado)"; continue; fi
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" 'uptime; ps -eo user:10,pcpu,comm --sort=-pcpu | head -4' || echo "(sem acesso a $h)"
    done
    echo "Confira: carga ~0,00 e nenhum processo de outros usuarios antes de medir."
}

etapa_serial() {
    msg "serial: corretude, calibracao, inversas de referencia e linha de base"
    local n primeiro=1
    # corretude e inversa de referencia da validacao (a primeira usa --verify: residuo A*A^-1)
    for n in $N_VALID; do
        if [ "$primeiro" = 1 ]; then
            ./programa_serial "$n" "m$n.bin" linha "inv$n.bin" "stats$n.txt" --verify | grep -E "A \* A|Elimin"
            primeiro=0
        else
            ./programa_serial "$n" "m$n.bin" linha "inv$n.bin" "st${n}v.txt" > /dev/null
        fi
    done
    # calibracao: tempo de eliminacao por N (cubico) para escolher o volume de dados
    for n in $CALIB_NS; do
        ./programa_serial "$n" "m$n.bin" linha /dev/null "stats$n.txt" > /dev/null
        echo "N=$n  $(grep -E 'eliminacao|total' "stats$n.txt" | tr '\n' ' ')"
    done
    # linha de base: UMA execucao, na maquina sem outros usuarios (node1). Demora cerca de 4 min.
    if [ -s "stats${N_FORTE}.txt" ] && [ "$FORCAR" != 1 ]; then
        echo "stats${N_FORTE}.txt ja existe; linha de base mantida (FORCAR=1 para refazer)"
    else
        echo "$(date +%T) linha de base N=$N_FORTE inicio" >> res/carga.log
        ./programa_serial "$N_FORTE" "m$N_FORTE.bin" linha "inv$N_FORTE.bin" "stats${N_FORTE}.txt" > /dev/null
        echo "$(date +%T) linha de base N=$N_FORTE fim" >> res/carga.log
    fi
    grep tempo_ "stats${N_FORTE}.txt"
}

etapa_validar() {
    msg "validacao contra a inversa do serial (elemento a elemento)"
    local n p ult
    {
        for n in $N_VALID; do
            for p in 1 2 4; do
                echo "=== N=$n P=$p"
                machinefile "$p" || return 1
                # shellcheck disable=SC2086
                mpirun $MPIRUN_OPTS -np "$p" $(mf_args "$p") ./programa_mpi "$n" "m$n.bin" --comparar "inv$n.bin"
            done
        done
    } > validacao_p124.txt 2>&1
    grep -E "===|VALIDACAO" validacao_p124.txt
    machinefile 4 || return 1
    # N de producao com P=4 (se a inversa de referencia existir)
    if [ -s "inv${N_FORTE}.bin" ]; then
        # shellcheck disable=SC2086
        mpirun $MPIRUN_OPTS -np 4 $(mf_args 4) ./programa_mpi "$N_FORTE" "m$N_FORTE.bin" --comparar "inv${N_FORTE}.bin" > "teste_N${N_FORTE}_P4.txt" 2>&1
        grep VALIDACAO "teste_N${N_FORTE}_P4.txt"
    fi
    # versao otimizada, no maior N de validacao
    ult=${N_VALID##* }
    # shellcheck disable=SC2086
    mpirun $MPIRUN_OPTS -np 4 $(mf_args 4) ./programa_otimizado "$ult" "m$ult.bin" --comparar "inv$ult.bin" > "teste_otimizado_N${ult}_P4.txt" 2>&1
    grep -E "VALIDACAO|Volume" "teste_otimizado_N${ult}_P4.txt"
}

etapa_forte() {
    msg "strong scaling: N=$N_FORTE, P em {$PS_FORTE}, $REPS execucoes por P"
    local p r
    for p in $PS_FORTE; do
        for r in $(seq 1 "$REPS"); do
            executar programa_mpi "$p" "$N_FORTE" "P$p r$r" "res/mpi_P${p}_r$r.out" || return 1
        done
    done
    grep -h "^RESULT" res/mpi_P*_r*.out
}

# serie de weak scaling: prefixo (W|L), lista "P:N"
serie_fraca() {
    local pref=$1 lista=$2 par p n r
    for par in $lista; do
        p=${par%%:*}
        n=${par#*:}
        for r in $(seq 1 "$REPS"); do
            executar programa_mpi "$p" "$n" "$pref P$p N$n r$r" "res/mpi_${pref}_P${p}_N${n}_r$r.out" || return 1
        done
    done
}

etapa_fraco() {
    msg "weak scaling (1): N ~ P^(1/3), trabalho por processo constante"
    serie_fraca W "$WEAK_CBRT"
    grep -h "^RESULT" res/mpi_W_P*_N*_r*.out
    msg "weak scaling (2): N = 500*P (definicao literal do enunciado)"
    serie_fraca L "$WEAK_LIT"
    grep -h "^RESULT" res/mpi_L_P*_N*_r*.out
}

etapa_otimizada() {
    msg "versao otimizada: N=$N_FORTE, P em {$PS_FORTE}, $REPS execucoes por P"
    local p r
    for p in $PS_FORTE; do
        for r in $(seq 1 "$REPS"); do
            executar programa_otimizado "$p" "$N_FORTE" "O P$p r$r" "res/otim_P${p}_r$r.out" || return 1
        done
    done
    grep -h "^RESULT" res/otim_P*_r*.out
}

# ---- profiling: so executa o que existir; NAO foi executado no nosso cluster ----
etapa_perfil() {
    msg "perfil (/usr/bin/time, mpiP, perf): executa apenas as ferramentas encontradas"
    local pmax lib
    pmax=$(echo $PS_FORTE | tr ' ' '\n' | sort -n | tail -1)
    # 2.2.2: /usr/bin/time -v no serial e no launcher (mede o mpirun, nao os ranks)
    if [ -x /usr/bin/time ]; then
        /usr/bin/time -v ./programa_serial "$N_FORTE" "m$N_FORTE.bin" linha /dev/null "stats${N_FORTE}_tv.txt" > serial_tv.out 2> serial_tv.time
        machinefile "$pmax" || return 1
        # shellcheck disable=SC2086
        /usr/bin/time -v mpirun $MPIRUN_OPTS -np "$pmax" $(mf_args "$pmax") ./programa_mpi "$N_FORTE" "m$N_FORTE.bin" > "res/time_mpirun_P$pmax.out" 2> "res/time_mpirun_P$pmax.time"
        grep -E "Elapsed|User time|System time|Maximum resident" serial_tv.time "res/time_mpirun_P$pmax.time"
    else
        echo "[perfil] /usr/bin/time ausente: nao executado (limitacao declarada no relatorio)"
    fi
    # 2.6: mpiP por LD_PRELOAD (variavel MPIP: -f diretorio, -k profundidade da pilha de chamadas)
    lib=${MPIP_LIB:-$(find /usr /opt "$HOME" -name 'libmpiP.so*' 2> /dev/null | head -1)}
    if [ -n "$lib" ]; then
        mkdir -p res/mpip
        machinefile "$pmax" || return 1
        # shellcheck disable=SC2086
        MPIP="-f $PWD/res/mpip -k 2" mpirun $MPIRUN_OPTS -np "$pmax" $(mf_args "$pmax") -x LD_PRELOAD="$lib" -x MPIP ./programa_mpi "$N_FORTE" "m$N_FORTE.bin" > "res/mpip_P$pmax.out" 2>&1
        ls res/mpip/*.mpiP 2> /dev/null
    else
        echo "[perfil] mpiP ausente (libmpiP.so nao encontrada): nao executado (limitacao declarada)"
    fi
    # 2.7: perf stat por no (um rank por no); exige perf e permissao em perf_event_paranoid
    if command -v perf > /dev/null 2>&1; then
        machinefile "$pmax" || return 1
        # shellcheck disable=SC2086
        mpirun $MPIRUN_OPTS -np "$pmax" $(mf_args "$pmax") bash -c \
            'perf stat -e task-clock,cycles,instructions,cache-references,cache-misses -o res/perf_rank${OMPI_COMM_WORLD_RANK}.txt ./programa_mpi '"$N_FORTE m$N_FORTE.bin"' > /dev/null 2>&1'
        cat res/perf_rank*.txt 2> /dev/null | head -40
    else
        echo "[perfil] perf ausente: nao executado (limitacao declarada)"
    fi
}

etapa_resumo() {
    msg "resumo do strong scaling (Tserial = tempo de eliminacao da linha de base)"
    local ts p
    ts=$(tserial)
    if [ -z "$ts" ]; then echo "stats${N_FORTE}.txt ausente: rode a etapa serial" >&2; return 1; fi
    echo "Tserial = $ts s"
    for p in $PS_FORTE; do
        ls res/mpi_P"${p}"_r*.out > /dev/null 2>&1 || continue
        grep -h '^RESULT' res/mpi_P"${p}"_r*.out | sed -E 's/.*tmax_total=([0-9.]+).*/\1/' |
            awk -v p="$p" -v ts="$ts" '{s += $1; ss += $1 * $1; n++} END {m = s / n; sd = (n > 1) ? sqrt((ss - n * m * m) / (n - 1)) : 0;
                 printf "P=%-3d n=%d  Tmax = %.3f +- %.3f s  speedup = %.3f  eficiencia = %.1f%%\n", p, n, m, sd, ts / m, 100 * ts / m / p}'
    done
}

etapa_consolidar() {
    msg "consolidar em dados_completos.txt"
    local f
    {
        for f in res/*.out; do [ -f "$f" ] && { echo "######## $f"; cat "$f"; }; done
        for f in res/carga.log stats*.txt st*v.txt validacao_p124.txt teste_*.txt mf/mf_*; do [ -f "$f" ] && { echo "######## $f"; cat "$f"; }; done
    } > dados_completos.txt
    echo "$(ls res/*.out 2> /dev/null | wc -l) saidas em res/ ; $(wc -l < dados_completos.txt) linhas em dados_completos.txt"
}

etapa_limpeza() {
    msg "verificacao final: processos remanescentes em cada no (esperado: 0)"
    local h
    for h in "${HOSTS_ARR[@]}"; do
        if [ "$SKIP_SSH" = 1 ]; then echo "$h: (SKIP_SSH=1: ignorado)"; continue; fi
        echo -n "$h: "
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" 'pgrep -fc "[p]rograma_otimizado|[p]rograma_mpi|[o]rted"' || true
    done
}

# ---------------------------------------------------------------- despacho
ETAPAS="$*"
[ "$ETAPAS" = "tudo" ] && ETAPAS="compilar matrizes ocupacao serial validar forte fraco otimizada perfil resumo consolidar limpeza"
echo "inicio: $(date '+%F %T') em $(hostname); pasta: $PWD"
for e in $ETAPAS; do
    case "$e" in
        ajuda) ajuda ;;
        compilar | matrizes | ocupacao | serial | validar | forte | fraco | otimizada | perfil | resumo | consolidar | limpeza)
            "etapa_$e" || { echo "ERRO na etapa '$e'; interrompendo." >&2; exit 1; } ;;
        *) echo "etapa desconhecida: '$e' (use ./run_all.sh ajuda)" >&2; exit 2 ;;
    esac
done
echo; echo "fim: $(date '+%F %T')"
