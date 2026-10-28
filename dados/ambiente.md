# Ambiente de teste (para a Seção 7 do relatório)

## Cluster
- Cluster UMBU (LaCAD, UEFS): 1 frontend (`umbu`, 172.16.112.7) e 13 nós de cômputo, `node1` a `node13` (10.0.0.11 a 10.0.0.23).
- Acesso por SSH; sem gerenciador de jobs. Execução com `mpirun` e `machinefile`; `/home` compartilhado entre os nós.
- Todas as medições: 1 processo por nó, nós `node1` a `nodeP`, `mpirun` lançado a partir do `node1`.

## Por nó (informado no material do cluster)
- Processador Intel Core i3-3240, 2 GiB de RAM, 80 GiB de disco.

## Medido por nós
- `nproc` = 4 CPUs lógicas por nó.
- Interface `eth0` do node1: 1000 Mb/s (Gigabit Ethernet).
- Latência ICMP node1 → node2, 20 pacotes: rtt mín/méd/máx/mdev = 0,154 / 0,169 / 0,244 / 0,018 ms; 0% de perda.
- Verificações de ocupação dos 13 nós feitas durante o trabalho (18/10 e 22/10; a de 22/10 foi anterior às séries de weak scaling literal e da otimizada): load average 0,00 / 0,00 / 0,00 e nenhum processo de outro usuário. A carga dos demais nós durante cada execução não foi registrada.
- Ao fim das execuções otimizadas: nenhum processo `programa_mpi`, `programa_otimizado` ou `orted` sobrando em nenhum dos 13 nós.

## Especificação pública do i3-3240 (Intel; NÃO medida nos nós; citar como fonte externa)
- 2 núcleos, 4 threads (Hyper-Threading), 3,4 GHz, 3 MB de cache L3.

## Não registrado (decisão do grupo)
- Versões de SO, kernel, gcc e OpenMPI; `lscpu` completo dos nós.
- Ferramentas ausentes no cluster: `/usr/bin/time`, mpiP e `perf`. Substituto para o tempo externo: `date` por execução (`res/carga.log`; resolução de 1 s).

## Compilação usada
- `mpicc -O2 -g -o programa_mpi programa_mpi.c` e `mpicc -O2 -g -o programa_otimizado programa_otimizado.c` (sem `-lm`).
- Serial: `gcc -O2 -g -o programa_serial programa_serial.c -lm` (o `-lm` é necessário por causa do `sqrt` na verificação opcional).
