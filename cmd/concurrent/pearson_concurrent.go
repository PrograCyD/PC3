//go:build algorithms
// +build algorithms

package main

/*
PEARSON (Concurrente, SOLO Item-Based, con filtrado + shrinkage)

Mejoras de calidad:
-------------------
1) Se descartan similitudes <= 0:
       if sim <= 0 { continue }
   → evita vecinos "inversos" que dañan el ranking.

2) Se aplica SHRINKAGE por número de co-ocurrencias (n):
       sim' = (n / (n + shrink)) * sim
   - shrink (λ) es un hiperparámetro: típicamente 10, 20, 50...
   - Penaliza pares con pocos usuarios en común → menos ruido,
     suele mejorar Precision@K / Recall@K / NDCG@K.

Concurrencia
------------
- ratings_ui.csv con filas (uIdx, iIdx, rating), ordenado por usuario.
- Se agrupan filas por usuario y se envían a un pool de workers por canal `jobs`.
- Cada worker recorre los pares (i,j) del usuario y llama a `updatePair`.
- Mapa shardeado:
      shard[k].m : map[i]map[j]*accIC
  donde el shard se elige solo por i (ítem base).
- No se necesita merge posterior: se recorre cada shard directo para Top-K.

Parámetros
----------
  --k=20
  --min_co=3
  --pct_users=100
  --pct_items=100
  --workers=8
  --shrink=20   (0 = sin shrinkage)

Entrada
-------
  artifacts/ratings_ui.csv

Salida
------
  artifacts/sim/item_topk_pearson_conc.csv
  artifacts/sim/item_pearson_conc_report.txt
*/

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"math"
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
	outItemTopK   = "artifacts/sim/item_topk_pearson_conc.csv"
	outItemReport = "artifacts/sim/item_pearson_conc_report.txt"
)

// ===== tipos comunes =====

type kv struct {
	j int
	s float64
}

type rating struct {
	i int
	r float64
}

// acumulador para Pearson item-item
type accIC struct {
	sumX, sumY, sumX2, sumY2, sumXY float64
	n                               int
}

// ===== utils =====

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
	m  map[int]map[int]*accIC // i -> j -> accIC
}

func newShards() [numShards]*shard {
	var s [numShards]*shard
	for i := range s {
		s[i] = &shard{m: make(map[int]map[int]*accIC)}
	}
	return s
}

// shard solo en función de i (ítem base)
func shardIndex(i int) int {
	return int(hash32(i) & (numShards - 1))
}

// actualización de un par (i,j) dentro del shard correspondiente
func updatePair(shards [numShards]*shard, ia, ib int, ra, rb float64) {
	if ia == ib {
		return
	}
	// canonicalizar (i<j) para no duplicar pares
	if ia > ib {
		ia, ib = ib, ia
		ra, rb = rb, ra
	}
	idx := shardIndex(ia)
	s := shards[idx]

	s.mu.Lock()
	m := s.m[ia]
	if m == nil {
		m = make(map[int]*accIC)
		s.m[ia] = m
	}
	t := m[ib]
	if t == nil {
		t = &accIC{}
		m[ib] = t
	}
	t.sumX += ra
	t.sumY += rb
	t.sumX2 += ra * ra
	t.sumY2 += rb * rb
	t.sumXY += ra * rb
	t.n++
	s.mu.Unlock()
}

// ===== algoritmo concurrente ITEM-BASED (Pearson) =====

func runItemBasedPearsonConcurrent(
	k, minCo, pctUsers, pctItems, workers, shrink int,
) (string, error) {
	t0 := time.Now()

	// abrir CSV de ratings
	f, err := os.Open(inTriplets)
	if err != nil {
		return "", err
	}
	defer f.Close()
	rd := csv.NewReader(bufio.NewReader(f))
	_, _ = rd.Read() // header

	// canal de trabajos: cada trabajo = ratings de UN usuario
	jobs := make(chan []rating, workers*2)

	// shards globales
	shards := newShards()

	var wg sync.WaitGroup
	wg.Add(workers)

	var pairsUpdated uint64
	var usersKept, tripletsOK uint64

	worker := func() {
		defer wg.Done()
		for items := range jobs {
			n := len(items)
			for a := 0; a < n; a++ {
				ia, ra := items[a].i, items[a].r
				for b := a + 1; b < n; b++ {
					ib, rb := items[b].i, items[b].r
					updatePair(shards, ia, ib, ra, rb)
					atomic.AddUint64(&pairsUpdated, 1)
				}
			}
		}
	}

	for w := 0; w < workers; w++ {
		go worker()
	}

	// lectura del CSV agrupando por usuario
	var lastU = -1
	items := make([]rating, 0, 128)

	emitUser := func() {
		if len(items) == 0 {
			return
		}
		cp := make([]rating, len(items))
		copy(cp, items)
		jobs <- cp
		items = items[:0]
		usersKept++
	}

	for {
		rec, er := rd.Read()
		if er != nil {
			if er.Error() == "EOF" {
				break
			}
			continue
		}
		u, _ := strconv.Atoi(rec[0])
		i, _ := strconv.Atoi(rec[1])
		r, _ := strconv.ParseFloat(rec[2], 64)

		// muestreo por usuario
		if !keepByPct(u, pctUsers) {
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

		// muestreo por ítem
		if !keepByPct(i, pctItems) {
			continue
		}

		items = append(items, rating{i: i, r: r})
		tripletsOK++
	}
	emitUser()  // último usuario
	close(jobs) // ya no hay más trabajos
	wg.Wait()   // esperamos a los workers
	t1 := time.Since(t0)

	// ===== Top-K por ítem (recorriendo shards, sin merges) =====
	out := make(map[int][]kv)
	var simsKept, lines uint64

	for _, s := range shards {
		s.mu.Lock()
		for i, m := range s.m {
			cands := make([]kv, 0, len(m))
			for j, t := range m {
				if t.n < minCo {
					continue
				}
				n := float64(t.n)
				num := t.sumXY - (t.sumX*t.sumY)/n
				denX := t.sumX2 - (t.sumX*t.sumX)/n
				denY := t.sumY2 - (t.sumY*t.sumY)/n
				if denX <= 0 || denY <= 0 {
					continue
				}
				sim := num / (math.Sqrt(denX) * math.Sqrt(denY))

				// 1) descartar similitudes <= 0 (no aportan para recomendar)
				if sim <= 0 {
					continue
				}

				// 2) shrinkage opcional por nº de co-ocurrencias
				if shrink > 0 {
					sim *= n / (n + float64(shrink))
				}

				if !math.IsNaN(sim) && !math.IsInf(sim, 0) {
					cands = append(cands, kv{j: j, s: sim})
				}
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
	t2 := time.Since(t0)

	// escribir CSV
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
	t3 := time.Since(t0)

	rep := fmt.Sprintf(
		`== PEARSON ITEM-BASED (concurrente, shardeado + shrinkage) ==
pct_users / pct_items   : %d%% / %d%%
Workers (goroutines)    : %d
Shards globales         : %d
Shrink (λ)              : %d

Usuarios usados aprox.  : %d
Tripletas leídas ok     : %d
Pares (i,j) acumulados  : %d
Similitudes retenidas   : %d
Líneas escritas (CSV)   : %d
Parámetros              : k=%d  min_co=%d

Tiempos:
  Lectura + envío jobs  : %s
  Top-K por ítem        : %s
  Escribir CSV          : %s
  TOTAL                 : %s

Salida CSV:
  %s
`,
		pctUsers, pctItems, workers, numShards, shrink,
		usersKept, tripletsOK, pairsUpdated, simsKept, lines, k, minCo,
		t1, t2-t1, t3-t2, t3,
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
	flag.IntVar(&minCo, "min_co", 3, "mínimo co-ocurrencias para considerar similitud")
	flag.IntVar(&pctUsers, "pct_users", 100, "% de usuarios a considerar (0-100)")
	flag.IntVar(&pctItems, "pct_items", 100, "% de ítems a considerar (0-100)")
	flag.IntVar(&workers, "workers", 8, "número de goroutines")
	flag.IntVar(&shrink, "shrink", 20, "parámetro de shrinkage (0 = sin shrinkage)")
	flag.Parse()

	if err := os.MkdirAll("artifacts/sim", 0o755); err != nil {
		panic(err)
	}

	rep, err := runItemBasedPearsonConcurrent(k, minCo, pctUsers, pctItems, workers, shrink)
	if err != nil {
		panic(err)
	}
	fmt.Print(rep)
}
