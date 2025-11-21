//go:build algorithms
// +build algorithms

package main

/*
COSENO (Concurrente, Item-Based, OPTIMIZADO + MEJOR CALIDAD)

Mejoras de precisión / métricas:
--------------------------------
1) Se eliminan similitudes negativas:
       if sim <= 0 { continue }
   Esto evita que vecinos "inversos" contaminen las predicciones.

2) Se aplica SHRINKAGE por número de co-ocurrencias (c):
       sim' = (c / (c + shrink)) * sim
   - shrink es un hiperparámetro (λ); típicamente 10, 20, 50...
   - Penaliza pares con pocos usuarios en común → menos ruido.

3) Normas ||i|| se calculan en un primer pase:
       norms[i] += r*r

4) Sharding global con 64 shards para reducir contención.

Flags:
  --k=20
  --min_co=3
  --pct_users=100
  --pct_items=100
  --workers=8
  --shrink=20   (0 = sin shrinkage)
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
	"time"
)

// ======== rutas =========

const (
	inTriplets    = "artifacts/ratings_ui.csv"
	outItemTopK   = "artifacts/sim/item_topk_cosine_conc.csv"
	outItemReport = "artifacts/sim/item_cosine_conc_report.txt"
)

// ======== estructuras =========

type acc struct {
	dot float64
	c   int
}

type kv struct {
	j int
	s float64
}

type rating struct {
	i int
	r float64
}

// ======== utilidades =========

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

// ======== Sharding =========

const numShards = 64

type shard struct {
	mu sync.Mutex
	m  map[int]map[int]*acc // i -> j -> acc
}

func newShards() [numShards]*shard {
	var s [numShards]*shard
	for i := range s {
		s[i] = &shard{m: make(map[int]map[int]*acc)}
	}
	return s
}

func shardIndex(i, j int) int {
	if i > j {
		i, j = j, i
	}
	h := hash32(i*73856093 ^ j*19349663)
	return int(h & (numShards - 1))
}

func updatePair(shards [numShards]*shard, ia, ib int, ra, rb float64) {
	if ia == ib {
		return
	}
	if ia > ib {
		ia, ib = ib, ia
		ra, rb = rb, ra
	}
	idx := shardIndex(ia, ib)
	s := shards[idx]

	s.mu.Lock()
	m := s.m[ia]
	if m == nil {
		m = make(map[int]*acc)
		s.m[ia] = m
	}
	t := m[ib]
	if t == nil {
		t = &acc{}
		m[ib] = t
	}
	t.dot += ra * rb
	t.c++
	s.mu.Unlock()
}

// ======== Algoritmo ITEM-BASED =========

func runItemBasedCosineConcurrent(
	k, minCo, pctUsers, pctItems, workers, shrink int,
) (string, error) {

	// ---- PRIMER PASO: Calcular normas ||i|| ----
	norms := make(map[int]float64)
	{
		f, err := os.Open(inTriplets)
		if err != nil {
			return "", err
		}
		rd := csv.NewReader(bufio.NewReader(f))
		_, _ = rd.Read() // header

		for {
			rec, er := rd.Read()
			if er != nil {
				break
			}
			u, _ := strconv.Atoi(rec[0])
			i, _ := strconv.Atoi(rec[1])
			r, _ := strconv.ParseFloat(rec[2], 64)

			if !keepByPct(u, pctUsers) {
				continue
			}
			if !keepByPct(i, pctItems) {
				continue
			}

			norms[i] += r * r
		}
		f.Close()
	}

	// ---- SEGUNDO PASO: Concurrente ----

	t0 := time.Now()

	f, err := os.Open(inTriplets)
	if err != nil {
		return "", err
	}
	defer f.Close()
	rd := csv.NewReader(bufio.NewReader(f))
	_, _ = rd.Read()

	jobs := make(chan []rating, workers*4)
	shards := newShards()

	var wg sync.WaitGroup
	wg.Add(workers)

	var usersKept, tripletsOK, pairsUpdated uint64

	worker := func() {
		defer wg.Done()
		for items := range jobs {
			n := len(items)
			for a := 0; a < n; a++ {
				ia, ra := items[a].i, items[a].r
				for b := a + 1; b < n; b++ {
					ib, rb := items[b].i, items[b].r
					updatePair(shards, ia, ib, ra, rb)
					pairsUpdated++
				}
			}
		}
	}

	for w := 0; w < workers; w++ {
		go worker()
	}

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
			break
		}
		u, _ := strconv.Atoi(rec[0])
		i, _ := strconv.Atoi(rec[1])
		r, _ := strconv.ParseFloat(rec[2], 64)

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

		if !keepByPct(i, pctItems) {
			continue
		}

		items = append(items, rating{i: i, r: r})
		tripletsOK++
	}

	emitUser()
	close(jobs)
	wg.Wait()
	t1 := time.Now()

	// ---- Fusionar shards ----
	global := make(map[int]map[int]*acc)
	for _, s := range shards {
		s.mu.Lock()
		for ia, m := range s.m {
			G := global[ia]
			if G == nil {
				G = make(map[int]*acc, len(m))
				global[ia] = G
			}
			for ib, t := range m {
				g := G[ib]
				if g == nil {
					G[ib] = &acc{dot: t.dot, c: t.c}
				} else {
					g.dot += t.dot
					g.c += t.c
				}
			}
		}
		s.mu.Unlock()
	}
	t2 := time.Now()

	// ---- Top-K (con filtro de negativos + shrinkage) ----
	out := make(map[int][]kv)
	var simsKept, lines uint64

	for i, m := range global {
		cands := make([]kv, 0, len(m))
		normI := math.Sqrt(norms[i])
		if normI == 0 {
			continue
		}

		for j, t := range m {
			if t.c < minCo {
				continue
			}

			normJ := math.Sqrt(norms[j])
			if normJ == 0 {
				continue
			}

			sim := t.dot / (normI * normJ)

			// 1) descartamos similitudes <= 0
			if sim <= 0 {
				continue
			}

			// 2) shrinkage opcional por # de co-ocurrencias
			if shrink > 0 {
				sim *= float64(t.c) / float64(t.c+shrink)
			}

			if !math.IsNaN(sim) && !math.IsInf(sim, 0) {
				cands = append(cands, kv{j: j, s: sim})
			}
		}

		cands = topK(cands, k)
		out[i] = cands
		simsKept += uint64(len(cands))
	}

	t3 := time.Now()

	// ---- CSV ----
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

	t4 := time.Now()

	rep := fmt.Sprintf(
		`== COSENO ITEM-BASED (concurrente optimizado + shrinkage) ==
pct_users / pct_items   : %d%% / %d%%
Workers (goroutines)    : %d
Shards globales         : %d
Shrink (λ)              : %d

Usuarios usados aprox.  : %d
Tripletas leídas ok     : %d
Pares (i,j) acumulados  : %d
Similitudes retenidas   : %d
Líneas escritas (CSV)   : %d

Tiempos:
  Paso 1: Calcular normas     : (incluido antes de t0)
  Lectura + envío jobs        : %s
  Fusionar shards             : %s
  Top-K por ítem              : %s
  Escribir CSV                : %s
  TOTAL                       : %s

Salida CSV:
  %s
`,
		pctUsers, pctItems, workers, numShards, shrink,
		usersKept, tripletsOK, pairsUpdated, simsKept, lines,
		t1.Sub(t0), t2.Sub(t1), t3.Sub(t2), t4.Sub(t3), t4.Sub(t0),
		outItemTopK,
	)

	_ = os.WriteFile(outItemReport, []byte(rep), 0o644)
	return rep, nil
}

// ========= main =========

func main() {
	var k, minCo int
	var pctUsers, pctItems int
	var workers int
	var shrink int

	flag.IntVar(&k, "k", 20, "Top-K vecinos por ítem")
	flag.IntVar(&minCo, "min_co", 3, "mínimo co-ocurrencias")
	flag.IntVar(&pctUsers, "pct_users", 100, "% usuarios")
	flag.IntVar(&pctItems, "pct_items", 100, "% ítems")
	flag.IntVar(&workers, "workers", 8, "número de goroutines")
	flag.IntVar(&shrink, "shrink", 20, "parámetro de shrinkage (0 = sin shrinkage)")
	flag.Parse()

	_ = os.MkdirAll("artifacts/sim", 0o755)

	rep, err := runItemBasedCosineConcurrent(k, minCo, pctUsers, pctItems, workers, shrink)
	if err != nil {
		panic(err)
	}
	fmt.Print(rep)
}
