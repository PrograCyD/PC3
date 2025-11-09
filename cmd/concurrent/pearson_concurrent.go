//go:build algorithms
// +build algorithms

package main

/*
PEARSON (Concurrente) — Item-Based y User-Based

Modo USER (CSR centrada por usuario):
- Entradas: artifacts/matrix_user_csr/{indptr.bin,indices.bin,data.bin}, con r' = r - mean(u)
- Pearson sobre r' es equivalente a coseno(r'), por lo que acumulamos:
    xy = Σ r'_u * r'_v,  x2 = Σ r'_u^2,  y2 = Σ r'_v^2,  c = |co-ratings|
  y sim(u,v) = xy / (sqrt(x2)*sqrt(y2)), si c >= min_co.
- Paralelización: dividir rango de items en bloques; cada worker genera (u->v->acc) local y luego se reduce globalmente.

Modo ITEM (tripletas sin centrar):
- Entrada: artifacts/ratings_ui.csv (uIdx,iIdx,rating)
- Pearson entre ítems i,j sobre usuarios comunes acumula:
    sumX=Σ r_ui, sumY=Σ r_uj, sumX2=Σ r_ui^2, sumY2=Σ r_uj^2, sumXY=Σ r_ui*r_uj, n
  y sim(i,j) = (sumXY - sumX*sumY/n) / ( sqrt(sumX2 - sumX^2/n) * sqrt(sumY2 - sumY^2/n) ), si n>=min_co y var>0.
- Paralelización: dividir usuarios en lotes; cada worker acumula (i->j->acc) local y luego se reduce.

Ambos modos:
- Muestra determinista por id (--pct_users, --pct_items) para controlar el tamaño.
- Worker-pool + channels (jobs/results); reducción en el hilo principal.
- Salida: CSV Top-K + reporte con tiempos y workers (para comparar scalability y speedup).

Parámetros:
  --mode=item|user
  --k=20
  --min_co=3
  --pct_users=100
  --pct_items=100
  --workers=8
*/

import (
	"bufio"
	"encoding/binary"
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

// ==== rutas/constantes ====
const (
	// ITEM-BASED
	inTriplets = "artifacts/ratings_ui.csv"

	// USER-BASED CSR (centrado por usuario)
	csrIndptrPath  = "artifacts/matrix_user_csr/indptr.bin"
	csrIndicesPath = "artifacts/matrix_user_csr/indices.bin"
	csrDataPath    = "artifacts/matrix_user_csr/data.bin"

	// Salidas
	outItemTopK   = "artifacts/sim/item_topk_pearson_conc.csv"
	outItemReport = "artifacts/sim/item_pearson_conc_report.txt"

	outUserTopK   = "artifacts/sim/user_topk_pearson_conc.csv"
	outUserReport = "artifacts/sim/user_pearson_conc_report.txt"
)

// ==== util común ====

type kv struct {
	j int
	s float64
}

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

// ==== USER-BASED (CSR centrada) PEARSON CONCURRENTE ====
// Nota: como data = r' (centrado por usuario), Pearson ≡ coseno(data)

type accUC struct { // acumulador user-user
	xy, x2, y2 float64
	c          int
}

func runUserBasedPearsonConcurrent(k, minCo, pctUsers, pctItems, workers int) (string, error) {
	t0 := time.Now()

	indptr := readInt64(csrIndptrPath)
	indices := readInt32(csrIndicesPath)
	data := readFloat32(csrDataPath)
	U := len(indptr) - 1

	// índice invertido item -> [(u, r')]
	maxI := 0
	for _, x := range indices {
		if int(x)+1 > maxI {
			maxI = int(x) + 1
		}
	}
	itemUsers := make([][]struct {
		u int
		r float64
	}, maxI)

	var pairsAdded uint64
	for u := 0; u < U; u++ {
		if !keepByPct(u, pctUsers) {
			continue
		}
		start, end := indptr[u], indptr[u+1]
		for p := start; p < end; p++ {
			i := int(indices[p])
			if !keepByPct(i, pctItems) {
				continue
			}
			itemUsers[i] = append(itemUsers[i], struct {
				u int
				r float64
			}{u: u, r: float64(data[p])})
			pairsAdded++
		}
	}
	t1 := time.Now()

	// Worker-pool sobre bloques de items
	type job struct{ lo, hi int }
	jobs := make(chan job, workers)
	type part struct{ m map[int]map[int]*accUC }
	results := make(chan part, workers)

	worker := func() {
		local := make(map[int]map[int]*accUC)
		for jb := range jobs {
			for i := jb.lo; i < jb.hi; i++ {
				users := itemUsers[i]
				n := len(users)
				for a := 0; a < n; a++ {
					ua, xa := users[a].u, users[a].r
					for b := a + 1; b < n; b++ {
						ub, xb := users[b].u, users[b].r
						m := local[ua]
						if m == nil {
							m = make(map[int]*accUC)
							local[ua] = m
						}
						t := m[ub]
						if t == nil {
							t = &accUC{}
							m[ub] = t
						}
						t.xy += xa * xb
						t.x2 += xa * xa
						t.y2 += xb * xb
						t.c++
					}
				}
			}
		}
		results <- part{m: local}
	}

	wg := sync.WaitGroup{}
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() { defer wg.Done(); worker() }()
	}

	const chunk = 1024
	for lo := 0; lo < maxI; lo += chunk {
		hi := lo + chunk
		if hi > maxI {
			hi = maxI
		}
		jobs <- job{lo: lo, hi: hi}
	}
	close(jobs)

	go func() { wg.Wait(); close(results) }()

	// Reduce global
	global := make(map[int]map[int]*accUC)
	var pairsUpdated uint64
	for part := range results {
		for ua, m := range part.m {
			G := global[ua]
			if G == nil {
				G = make(map[int]*accUC, len(m))
				global[ua] = G
			}
			for ub, t := range m {
				g := G[ub]
				if g == nil {
					G[ub] = &accUC{xy: t.xy, x2: t.x2, y2: t.y2, c: t.c}
				} else {
					g.xy += t.xy
					g.x2 += t.x2
					g.y2 += t.y2
					g.c += t.c
				}
				pairsUpdated++
			}
		}
	}
	t2 := time.Now()

	// Top-K por usuario (Pearson sobre r' ≡ coseno(r'))
	out := make(map[int][]kv)
	var simsKept, lines uint64
	for u, m := range global {
		cands := make([]kv, 0, len(m))
		for v, t := range m {
			if t.c < minCo || t.x2 == 0 || t.y2 == 0 {
				continue
			}
			sim := t.xy / (math.Sqrt(t.x2) * math.Sqrt(t.y2))
			if !math.IsNaN(sim) && !math.IsInf(sim, 0) {
				cands = append(cands, kv{j: v, s: sim})
			}
		}
		cands = topK(cands, k)
		out[u] = cands
		simsKept += uint64(len(cands))
	}

	err := writeTopKCSV(outUserTopK, []string{"uIdx", "vIdx", "sim"}, func(write func([]string)) {
		for u, list := range out {
			for _, p := range list {
				write([]string{strconv.Itoa(u), strconv.Itoa(p.j), fmt.Sprintf("%.6f", p.s)})
				lines++
			}
		}
	})
	if err != nil {
		return "", err
	}
	t3 := time.Now()

	rep := fmt.Sprintf(
		`== PEARSON USER-BASED (concurrente) ==
Usuarios totales (U)     : %d
Ítems totales (I) aprox. : %d
pct_users / pct_items    : %d%% / %d%%
Workers (goroutines)     : %d

Pares u-i añadidos aprox.: %d
Pares (u,v) acumulados   : %d
Similitudes retenidas    : %d
Líneas escritas (CSV)    : %d
Parámetros               : k=%d  min_co=%d

Tiempos:
  Invertir (item->users) : %s
  Workers (acumular)     : %s
  Top-K por usuario      : %s
  TOTAL                  : %s

Salida CSV:
  %s
`, U, maxI, pctUsers, pctItems, workers,
		pairsAdded, pairsUpdated, simsKept, lines, k, minCo,
		t1.Sub(t0), t2.Sub(t1), t3.Sub(t2), t3.Sub(t0),
		outUserTopK)

	if err := os.WriteFile(outUserReport, []byte(rep), 0o644); err != nil {
		return "", err
	}
	return rep, nil
}

// ==== ITEM-BASED (tripletas sin centrar) PEARSON CONCURRENTE ====
// Acumula sumX,sumY,sumX2,sumY2,sumXY,n por (i,j), y luego corr(i,j).

type accIC struct { // acumulador item-item
	sumX, sumY, sumX2, sumY2, sumXY float64
	n                               int
}

func runItemBasedPearsonConcurrent(k, minCo, pctUsers, pctItems, workers int) (string, error) {
	t0 := time.Now()

	f, err := os.Open(inTriplets)
	if err != nil {
		return "", err
	}
	defer f.Close()
	rd := csv.NewReader(bufio.NewReader(f))
	_, _ = rd.Read() // header

	type rating struct {
		i int
		r float64
	}
	type userBlock struct{ users [][]rating }
	jobs := make(chan userBlock, workers)
	type part struct{ m map[int]map[int]*accIC }
	results := make(chan part, workers)

	worker := func() {
		local := make(map[int]map[int]*accIC)
		for blk := range jobs {
			for _, items := range blk.users {
				for a := 0; a < len(items); a++ {
					ia, ra := items[a].i, items[a].r
					for b := a + 1; b < len(items); b++ {
						ib, rb := items[b].i, items[b].r
						m := local[ia]
						if m == nil {
							m = make(map[int]*accIC)
							local[ia] = m
						}
						t := m[ib]
						if t == nil {
							t = &accIC{}
							m[ib] = t
						}
						// acumular para Pearson (no centrado)
						t.sumX += ra
						t.sumY += rb
						t.sumX2 += ra * ra
						t.sumY2 += rb * rb
						t.sumXY += ra * rb
						t.n++
					}
				}
			}
		}
		results <- part{m: local}
	}

	wg := sync.WaitGroup{}
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() { defer wg.Done(); worker() }()
	}

	// lectura y batching
	var lastU = -1
	var items []rating
	const usersPerBlock = 4096
	block := userBlock{users: make([][]rating, 0, usersPerBlock)}
	var usersKept, tripletsOK uint64

	emitUser := func() {
		if len(items) == 0 {
			return
		}
		block.users = append(block.users, append([]rating(nil), items...))
		if len(block.users) >= usersPerBlock {
			jobs <- block
			block = userBlock{users: make([][]rating, 0, usersPerBlock)}
		}
		items = items[:0]
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

		if !keepByPct(u, pctUsers) {
			if lastU != -1 && u != lastU {
				emitUser()
				lastU = u
			}
			continue
		}
		if lastU == -1 {
			lastU = u
		}
		if u != lastU {
			emitUser()
			lastU = u
			usersKept++
		}

		if !keepByPct(i, pctItems) {
			continue
		}
		items = append(items, rating{i: i, r: r})
		tripletsOK++
	}
	emitUser()
	if len(block.users) > 0 {
		jobs <- block
	}
	close(jobs)

	go func() { wg.Wait(); close(results) }()

	// reduce global
	global := make(map[int]map[int]*accIC)
	var pairsUpdated, simsKept, lines uint64
	for part := range results {
		for ia, m := range part.m {
			G := global[ia]
			if G == nil {
				G = make(map[int]*accIC, len(m))
				global[ia] = G
			}
			for ib, t := range m {
				g := G[ib]
				if g == nil {
					G[ib] = &accIC{
						sumX: t.sumX, sumY: t.sumY, sumX2: t.sumX2, sumY2: t.sumY2, sumXY: t.sumXY, n: t.n,
					}
				} else {
					g.sumX += t.sumX
					g.sumY += t.sumY
					g.sumX2 += t.sumX2
					g.sumY2 += t.sumY2
					g.sumXY += t.sumXY
					g.n += t.n
				}
				pairsUpdated++
			}
		}
	}
	t1 := time.Now()

	// Top-K por ítem (Pearson)
	out := make(map[int][]kv)
	for i, m := range global {
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
			if !math.IsNaN(sim) && !math.IsInf(sim, 0) {
				cands = append(cands, kv{j: j, s: sim})
			}
		}
		cands = topK(cands, k)
		out[i] = cands
		simsKept += uint64(len(cands))
	}

	// escribir CSV
	err = writeTopKCSV(outItemTopK, []string{"iIdx", "jIdx", "sim"}, func(write func([]string)) {
		for i, list := range out {
			for _, p := range list {
				write([]string{strconv.Itoa(i), strconv.Itoa(p.j), fmt.Sprintf("%.6f", p.s)})
				lines++
			}
		}
	})
	if err != nil {
		return "", err
	}
	t2 := time.Now()

	rep := fmt.Sprintf(
		`== PEARSON ITEM-BASED (concurrente) ==
pct_users / pct_items   : %d%% / %d%%
Workers (goroutines)    : %d

Usuarios usados aprox.  : %d
Tripletas leídas ok     : %d
Pares (i,j) acumulados  : %d
Similitudes retenidas   : %d
Líneas escritas (CSV)   : %d
Parámetros              : k=%d  min_co=%d

Tiempos:
  Workers (acumular)    : %s
  Top-K por ítem        : %s
  Escribir CSV          : %s
  TOTAL                 : %s

Salida CSV:
  %s
`, pctUsers, pctItems, workers,
		usersKept, tripletsOK, pairsUpdated, simsKept, lines, k, minCo,
		t1.Sub(t0), t2.Sub(t1), time.Since(t2), time.Since(t0),
		outItemTopK)

	if err := os.WriteFile(outItemReport, []byte(rep), 0o644); err != nil {
		return "", err
	}
	return rep, nil
}

// ==== MAIN selector ====

func main() {
	var mode string
	var k, minCo int
	var pctUsers, pctItems int
	var workers int

	flag.StringVar(&mode, "mode", "item", "item | user")
	flag.IntVar(&k, "k", 20, "Top-K vecinos")
	flag.IntVar(&minCo, "min_co", 3, "mínimo co-ocurrencias")
	flag.IntVar(&pctUsers, "pct_users", 100, "% de usuarios a considerar (0-100)")
	flag.IntVar(&pctItems, "pct_items", 100, "% de ítems a considerar (0-100)")
	flag.IntVar(&workers, "workers", 8, "número de goroutines/ workers")
	flag.Parse()

	if err := os.MkdirAll("artifacts/sim", 0o755); err != nil {
		panic(err)
	}

	var rep string
	var err error
	switch mode {
	case "item":
		rep, err = runItemBasedPearsonConcurrent(k, minCo, pctUsers, pctItems, workers)
	case "user":
		rep, err = runUserBasedPearsonConcurrent(k, minCo, pctUsers, pctItems, workers)
	default:
		panic("modo inválido: use --mode=item | --mode=user")
	}
	if err != nil {
		panic(err)
	}
	fmt.Print(rep)
}

// ==== util lectura CSR ====

func readInt64(path string) []int64 {
	b, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	n := len(b) / 8
	out := make([]int64, n)
	for i := 0; i < n; i++ {
		out[i] = int64(binary.LittleEndian.Uint64(b[i*8:]))
	}
	return out
}
func readInt32(path string) []int32 {
	b, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	n := len(b) / 4
	out := make([]int32, n)
	for i := 0; i < n; i++ {
		out[i] = int32(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}
func readFloat32(path string) []float32 {
	b, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}
