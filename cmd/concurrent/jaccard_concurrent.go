//go:build algorithms
// +build algorithms

package main

/*
JACCARD (Concurrente, SOLO Item-Based, optimizado)

Resumen
-------
- Trabaja con artifacts/ratings_ui.csv con filas: (uIdx, iIdx, rating), ordenado por usuario.
- Para cada usuario u, consideramos el conjunto de ítems que ha valorado:
    U(u) = { i | (u,i) en ratings }

- Para cada par de ítems (i,j):
    inter(i,j) = | U(i) ∩ U(j) |
    |U(i)|     = número de usuarios que valoran i
    |U(j)|     = número de usuarios que valoran j
    union      = |U(i)| + |U(j)| - inter(i,j)

    Jaccard(i,j) = inter / union

- Filtro:
    inter >= min_co
    union > 0

- Shrinkage opcional:
    sim_shrunk = Jaccard * inter / (inter + shrink)
  para penalizar pares con pocas co-ocurrencias.

Concurrencia / performance
--------------------------
- PASO 1 (secuencial, rápido): una pasada al CSV para contar |U(i)| (itemCount[i]).
- PASO 2 (concurrente):
    * Se vuelve a leer el CSV agrupando las filas por usuario (canastas).
    * Cada canasta (slice de ítems) se envía por canal jobs a un pool de workers.
    * Cada worker recorre todos los pares (i,j) de la canasta y llama a updatePair:
        - Tenemos numShards shards globales.
        - Cada shard tiene map[i]map[j]*accJ + sync.Mutex.
        - El shard se escoge por hash(i,j) ⇒ balance de carga y poca contención.
    * No hay mapas locales ni fase de reduce costosa: todo se acumula en los shards.

Parámetros
----------
  --k=20            Top-K vecinos por ítem
  --min_co=3        mínimo de co-ocurrencias (inter) para considerar similitud
  --pct_users=100   % de usuarios (muestreo determinista por uIdx)
  --pct_items=100   % de ítems (muestreo determinista por iIdx)
  --workers=8       número de goroutines
  --shrink=0        shrinkage para Jaccard (0 = sin shrink)

Entradas
--------
  artifacts/ratings_ui.csv   (uIdx,iIdx,rating)

Salidas
-------
  artifacts/sim/item_topk_jaccard_conc.csv
  artifacts/sim/item_jaccard_conc_report.txt
*/

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// ===== rutas de entrada/salida =====

const (
	inTriplets    = "artifacts/ratings_ui.csv"
	outItemTopK   = "artifacts/sim/item_topk_jaccard_conc.csv"
	outItemReport = "artifacts/sim/item_jaccard_conc_report.txt"
)

// ===== tipos comunes =====

type kv struct {
	j int
	s float64
}

// acumulador para Jaccard item-item (solo intersección)
type accJ struct {
	inter int
}

// ===== utilidades =====

func hash32(x int) uint32 {
	h := uint32(2166136261)
	v := uint32(x)
	for k := 0; k < 4; k++ {
		h ^= (v >> (8 * uint(k))) & 0xff
		h *= 16777619
	}
	return h
}

func keepByPct(id int, pct int) bool {
	if pct >= 100 {
		return true
	}
	if pct <= 0 {
		return false
	}
	return int(hash32(id)%100) < pct
}

func topK(list []kv, k int) []kv {
	sort.Slice(list, func(a, b int) bool { return list[a].s > list[b].s })
	if len(list) > k {
		return list[:k]
	}
	return list
}

func writeTopKCSV(path string, header []string, rows func(write func([]string))) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(bufio.NewWriter(f))
	defer w.Flush()

	_ = w.Write(header)
	rows(func(rec []string) { _ = w.Write(rec) })
	return nil
}

// ===== estructura shardeada =====

// potencia de 2 para usar & en vez de %
const numShards = 64

type shard struct {
	mu sync.Mutex
	m  map[int]map[int]*accJ // i -> j -> accJ
}

func newShards() [numShards]*shard {
	var s [numShards]*shard
	for i := range s {
		s[i] = &shard{m: make(map[int]map[int]*accJ)}
	}
	return s
}

// shardIndex en función de (i,j) (canonizado i<j)
func shardIndex(i, j int) int {
	if i > j {
		i, j = j, i
	}
	h := hash32(i*73856093 ^ j*19349663)
	return int(h & (numShards - 1))
}

// actualización de un par (i,j) dentro del shard correspondiente
func updatePair(shards [numShards]*shard, ia, ib int) {
	if ia == ib {
		return
	}
	// canonicalizar (i<j) para no duplicar pares
	if ia > ib {
		ia, ib = ib, ia
	}
	idx := shardIndex(ia, ib)
	s := shards[idx]

	s.mu.Lock()
	m := s.m[ia]
	if m == nil {
		m = make(map[int]*accJ)
		s.m[ia] = m
	}
	t := m[ib]
	if t == nil {
		t = &accJ{}
		m[ib] = t
	}
	t.inter++
	s.mu.Unlock()
}

// ===== algoritmo concurrente ITEM-BASED (Jaccard) =====

func runItemBasedJaccardConcurrent(k, minCo, pctUsers, pctItems, workers, shrink int) (string, error) {
	t0 := time.Now()

	// === PASO 1: contar |U(i)| por ítem (itemCount[i]) ===
	itemCount := make(map[int]int)

	f1, err := os.Open(inTriplets)
	if err != nil {
		return "", err
	}
	rd1 := csv.NewReader(bufio.NewReader(f1))
	_, _ = rd1.Read() // header

	var tripletsCount uint64

	for {
		rec, er := rd1.Read()
		if er != nil {
			break
		}
		u, _ := strconv.Atoi(rec[0])
		i, _ := strconv.Atoi(rec[1])

		if !keepByPct(u, pctUsers) || !keepByPct(i, pctItems) {
			continue
		}
		itemCount[i]++
		tripletsCount++
	}
	f1.Close()
	tCount := time.Since(t0)

	// === PASO 2: worker-pool + shards (pares por usuario) ===

	f2, err := os.Open(inTriplets)
	if err != nil {
		return "", err
	}
	defer f2.Close()
	rd2 := csv.NewReader(bufio.NewReader(f2))
	_, _ = rd2.Read() // header

	jobs := make(chan []int, workers*4)
	shards := newShards()

	var wg sync.WaitGroup
	wg.Add(workers)

	var pairsUpdated uint64
	var usersKept uint64

	worker := func() {
		defer wg.Done()
		for basket := range jobs {
			n := len(basket)
			for a := 0; a < n; a++ {
				ia := basket[a]
				for b := a + 1; b < n; b++ {
					ib := basket[b]
					updatePair(shards, ia, ib)
					atomic.AddUint64(&pairsUpdated, 1)
				}
			}
		}
	}

	for w := 0; w < workers; w++ {
		go worker()
	}

	// lectura agrupando por usuario
	var lastU = -1
	basket := make([]int, 0, 64)

	emitUser := func() {
		if len(basket) == 0 {
			return
		}
		cp := make([]int, len(basket))
		copy(cp, basket)
		jobs <- cp
		basket = basket[:0]
		atomic.AddUint64(&usersKept, 1)
	}

	for {
		rec, er := rd2.Read()
		if er != nil {
			break
		}
		u, _ := strconv.Atoi(rec[0])
		i, _ := strconv.Atoi(rec[1])

		// aplicar los mismos filtros de muestreo
		if !keepByPct(u, pctUsers) || !keepByPct(i, pctItems) {
			// cerrar canasta si cambiamos de usuario
			if lastU != -1 && u != lastU {
				emitUser()
				lastU = u
			}
			continue
		}

		if lastU == -1 {
			lastU = u
		} else if u != lastU {
			emitUser()
			lastU = u
		}
		basket = append(basket, i)
	}
	emitUser()  // último usuario
	close(jobs) // no más trabajos
	wg.Wait()   // esperar a todos los workers
	tPairs := time.Since(t0) - tCount

	// === PASO 3: calcular Jaccard (con shrink) y Top-K por ítem ===

	out := make(map[int][]kv)
	var simsKept, lines uint64

	for _, s := range shards {
		s.mu.Lock()
		for i, m := range s.m {
			cands := make([]kv, 0, len(m))
			countI := itemCount[i]
			if countI == 0 {
				continue
			}
			for j, t := range m {
				if t.inter < minCo {
					continue
				}
				countJ := itemCount[j]
				if countJ == 0 {
					continue
				}
				union := countI + countJ - t.inter
				if union <= 0 {
					continue
				}
				sim := float64(t.inter) / float64(union)
				if sim <= 0 {
					continue
				}
				// Shrinkage: sim_shrunk = sim * inter / (inter + shrink)
				if shrink > 0 {
					w := float64(t.inter) / (float64(t.inter) + float64(shrink))
					sim *= w
				}
				cands = append(cands, kv{j: j, s: sim})
			}
			if len(cands) == 0 {
				continue
			}
			cands = topK(cands, k)
			out[i] = cands
			simsKept += uint64(len(cands))
		}
		s.mu.Unlock()
	}
	tTop := time.Since(t0) - tCount - tPairs

	// === PASO 4: escribir CSV ===

	err = writeTopKCSV(outItemTopK, []string{"iIdx", "jIdx", "sim"}, func(write func([]string)) {
		for i, list := range out {
			for _, p := range list {
				write([]string{
					strconv.Itoa(i),
					strconv.Itoa(p.j),
					fmt.Sprintf("%.6f", p.s),
				})
				lines++
			}
		}
	})
	if err != nil {
		return "", err
	}
	tCSV := time.Since(t0) - tCount - tPairs - tTop

	total := time.Since(t0)

	rep := fmt.Sprintf(
		`== JACCARD ITEM-BASED (concurrente, shardeado) ==
pct_users / pct_items   : %d%% / %d%%
Workers (goroutines)    : %d
Shards globales         : %d
Shrink                  : %d

Usuarios usados aprox.  : %d
Tripletas leídas (paso1): %d
Pares (i,j) acumulados  : %d
Similitudes retenidas   : %d
Líneas escritas (CSV)   : %d
Parámetros              : k=%d  min_co=%d

Tiempos:
  Paso 1: contar |U(i)|        : %s
  Paso 2: workers (pares)      : %s
  Paso 3: Top-K por ítem       : %s
  Paso 4: Escribir CSV         : %s
  TOTAL                        : %s

Salida CSV:
  %s
`,
		pctUsers, pctItems, workers, numShards, shrink,
		usersKept, tripletsCount, pairsUpdated, simsKept, lines, k, minCo,
		tCount, tPairs, tTop, tCSV, total,
		outItemTopK,
	)

	if err := os.WriteFile(outItemReport, []byte(rep), 0o644); err != nil {
		return "", err
	}
	return rep, nil
}

// ========= main =========

func main() {
	var k, minCo int
	var pctUsers, pctItems int
	var workers int
	var shrink int

	flag.IntVar(&k, "k", 20, "Top-K vecinos por ítem")
	flag.IntVar(&minCo, "min_co", 3, "mínimo co-ocurrencias (inter)")
	flag.IntVar(&pctUsers, "pct_users", 100, "% de usuarios a considerar (0-100)")
	flag.IntVar(&pctItems, "pct_items", 100, "% de ítems a considerar (0-100)")
	flag.IntVar(&workers, "workers", 8, "número de goroutines")
	flag.IntVar(&shrink, "shrink", 0, "shrinkage para Jaccard (0 = sin shrink)")
	flag.Parse()

	if err := os.MkdirAll("artifacts/sim", 0o755); err != nil {
		panic(err)
	}

	rep, err := runItemBasedJaccardConcurrent(k, minCo, pctUsers, pctItems, workers, shrink)
	if err != nil {
		panic(err)
	}
	fmt.Print(rep)
}
