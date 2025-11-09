//go:build algorithms
// +build algorithms

package main

/*
COSENO (Concurrente) — Item-Based y User-Based

Descripción general:
- Modo ITEM: Usa artifacts/ratings_ui.csv (uIdx,iIdx,rating), agrupa por usuario y acumula productos
  para pares (i,j). Normaliza con ||i|| y ||j|| → sim_cos(i,j) = dot(i,j) / (||i||·||j||).
- Modo USER: Usa la CSR centrada por usuario (indptr/indices/data con r' = r - mean_u). Construye
  índice invertido item -> [(u, r')], acumula productos para pares (u,v). Normaliza con ||u|| y ||v||.

Concurrencia:
- Divide la matriz en subproblemas:
  * ITEM-BASED: Lotes de usuarios (subfilas) enviados por channel a workers (worker-pool).
  * USER-BASED: Bloques de ítems (subcolumnas) enviados por channel a workers.
- Cada worker usa mapas locales (sin locks). El hilo principal fusiona (reduce) al final.
- Canales:
  * jobs   : trabajos (rangos / lotes) para procesar
  * result : resultados parciales (mapas de acumuladores) a fusionar
- Patrones: worker-pool + reduce.

Parámetros:
  --mode=item|user
  --k=20
  --min_co=3
  --pct_users=100   (0-100, muestreo determinista por uIdx)
  --pct_items=100   (0-100, muestreo determinista por iIdx)
  --workers=8       (número de goroutines)

Entradas:
  * mode=item:
      - artifacts/ratings_ui.csv
  * mode=user:
      - artifacts/matrix_user_csr/indptr.bin
      - artifacts/matrix_user_csr/indices.bin
      - artifacts/matrix_user_csr/data.bin

Salidas:
  * mode=item:
      - artifacts/sim/item_topk_cosine_conc.csv
      - artifacts/sim/item_cosine_conc_report.txt
  * mode=user:
      - artifacts/sim/user_topk_cosine_conc.csv
      - artifacts/sim/user_cosine_conc_report.txt
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

// ==== rutas por modo ====
const (
	// ITEM-BASED (triplets)
	inTriplets = "artifacts/ratings_ui.csv"

	// USER-BASED (CSR centrada por usuario)
	csrIndptrPath  = "artifacts/matrix_user_csr/indptr.bin"
	csrIndicesPath = "artifacts/matrix_user_csr/indices.bin"
	csrDataPath    = "artifacts/matrix_user_csr/data.bin"

	// Salidas
	outItemTopK   = "artifacts/sim/item_topk_cosine_conc.csv"
	outItemReport = "artifacts/sim/item_cosine_conc_report.txt"

	outUserTopK   = "artifacts/sim/user_topk_cosine_conc.csv"
	outUserReport = "artifacts/sim/user_cosine_conc_report.txt"
)

// ==== util común ====

type acc struct {
	dot, n2a, n2b float64
	c             int
}

type kv struct {
	j int
	s float64
}

type pair struct {
	a, b int
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

// ==== USER-BASED (CSR) CONCURRENTE ====
// Divide el rango de items en bloques; cada worker:
//   - toma items [lo,hi)
//   - recorre lista de usuarios del ítem (u, r')
//   - acumula pares (u,v) en mapa local
// Al final, se fusionan mapas y se saca Top-K por usuario.

func runUserBasedCosineConcurrent(k, minCo, pctUsers, pctItems, workers int) (rep string, err error) {
	t0 := time.Now()

	// Cargar CSR
	indptr := readInt64(csrIndptrPath)
	indices := readInt32(csrIndicesPath)
	data := readFloat32(csrDataPath)
	U := len(indptr) - 1

	// Construir índice invertido item -> [(u, r')]
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

	// Trabajos: partir items en bloques
	type job struct{ lo, hi int }
	jobs := make(chan job, workers)
	type part struct {
		m map[int]map[int]*acc // u -> v -> acc
	}
	results := make(chan part, workers)

	worker := func() {
		local := make(map[int]map[int]*acc)
		for jb := range jobs {
			for i := jb.lo; i < jb.hi; i++ {
				users := itemUsers[i]
				n := len(users)
				for a := 0; a < n; a++ {
					ua, xa := users[a].u, users[a].r
					for b := a + 1; b < n; b++ {
						ub, xb := users[b].u, users[b].r
						// aplicar filtros por pctUsers ya hechos arriba (evita re-check)
						m := local[ua]
						if m == nil {
							m = make(map[int]*acc)
							local[ua] = m
						}
						t := m[ub]
						if t == nil {
							t = &acc{}
							m[ub] = t
						}
						t.dot += xa * xb
						t.n2a += xa * xa
						t.n2b += xb * xb
						t.c++
					}
				}
			}
		}
		results <- part{m: local}
	}

	// lanzar workers
	wg := sync.WaitGroup{}
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			worker()
		}()
	}

	// encolar bloques de items
	const chunk = 1024
	for lo := 0; lo < maxI; lo += chunk {
		hi := lo + chunk
		if hi > maxI {
			hi = maxI
		}
		jobs <- job{lo: lo, hi: hi}
	}
	close(jobs)
	// recoger resultados
	go func() {
		wg.Wait()
		close(results)
	}()

	// fusionar mapas
	global := make(map[int]map[int]*acc)
	var pairsUpdated uint64
	for part := range results {
		for ua, m := range part.m {
			G := global[ua]
			if G == nil {
				G = make(map[int]*acc, len(m))
				global[ua] = G
			}
			for ub, t := range m {
				g := G[ub]
				if g == nil {
					G[ub] = &acc{dot: t.dot, n2a: t.n2a, n2b: t.n2b, c: t.c}
				} else {
					g.dot += t.dot
					g.n2a += t.n2a
					g.n2b += t.n2b
					g.c += t.c
				}
				pairsUpdated++
			}
		}
	}
	t2 := time.Now()

	// Top-K por usuario (sim cos)
	out := make(map[int][]kv)
	var simsKept, lines uint64
	for u, m := range global {
		cands := make([]kv, 0, len(m))
		for v, t := range m {
			if t.c < minCo || t.n2a == 0 || t.n2b == 0 {
				continue
			}
			sim := t.dot / (math.Sqrt(t.n2a) * math.Sqrt(t.n2b))
			if !math.IsNaN(sim) && !math.IsInf(sim, 0) {
				cands = append(cands, kv{j: v, s: sim})
			}
		}
		cands = topK(cands, k)
		out[u] = cands
		simsKept += uint64(len(cands))
	}
	t3 := time.Now()

	// Escribir CSV
	err = writeTopKCSV(outUserTopK, []string{"uIdx", "vIdx", "sim"}, func(write func([]string)) {
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
	t4 := time.Now()

	rep = fmt.Sprintf(
		`== COSENO USER-BASED (concurrente) ==
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
  Escribir CSV           : %s
  TOTAL                  : %s

Salida CSV:
  %s
`, U, maxI, pctUsers, pctItems, workers,
		pairsAdded, pairsUpdated, simsKept, lines, k, minCo,
		t1.Sub(t0), t2.Sub(t1), t3.Sub(t2), t4.Sub(t3), t4.Sub(t0),
		outUserTopK)

	if err := os.WriteFile(outUserReport, []byte(rep), 0o644); err != nil {
		return "", err
	}
	return rep, nil
}

// ==== ITEM-BASED (triplets) CONCURRENTE ====
// Lee ratings_ui.csv (ordenado por uIdx), acumula por usuario en lotes:
//   - Cada lote (varios usuarios contiguos) va a un worker.
//   - El worker arma mapa local i -> j -> acc.
//   - Se fusiona globalmente y luego se hace Top-K por ítem.

func runItemBasedCosineConcurrent(k, minCo, pctUsers, pctItems, workers int) (rep string, err error) {
	t0 := time.Now()

	// lector del CSV
	f, err := os.Open(inTriplets)
	if err != nil {
		return "", err
	}
	defer f.Close()
	rd := csv.NewReader(bufio.NewReader(f))
	_, _ = rd.Read() // header

	// buffers por usuario actual
	type rating struct {
		i int
		r float64
	}
	type userBlock struct {
		users [][]rating // bloque de varios usuarios
	}
	jobs := make(chan userBlock, workers)
	type part struct {
		m map[int]map[int]*acc // i -> j -> acc
	}
	results := make(chan part, workers)

	// worker
	worker := func() {
		local := make(map[int]map[int]*acc)
		for blk := range jobs {
			for _, items := range blk.users {
				// acumular pares (i,j) dentro del usuario
				for a := 0; a < len(items); a++ {
					ia, ra := items[a].i, items[a].r
					for b := a + 1; b < len(items); b++ {
						ib, rb := items[b].i, items[b].r
						m := local[ia]
						if m == nil {
							m = make(map[int]*acc)
							local[ia] = m
						}
						t := m[ib]
						if t == nil {
							t = &acc{}
							m[ib] = t
						}
						t.dot += ra * rb
						t.n2a += ra * ra
						t.n2b += rb * rb
						t.c++
					}
				}
			}
		}
		results <- part{m: local}
	}

	// lanzar workers
	wg := sync.WaitGroup{}
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			worker()
		}()
	}

	// lectura y batching por usuarios
	var lastU = -1
	var items []rating
	const usersPerBlock = 4096
	block := userBlock{users: make([][]rating, 0, usersPerBlock)}

	var usersKept, tripletsOK, pairsUpdated uint64
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
	emitUser() // último usuario del archivo
	if len(block.users) > 0 {
		jobs <- block
	}
	close(jobs)

	// recolectar resultados
	go func() {
		wg.Wait()
		close(results)
	}()
	global := make(map[int]map[int]*acc)
	for part := range results {
		for ia, m := range part.m {
			G := global[ia]
			if G == nil {
				G = make(map[int]*acc, len(m))
				global[ia] = G
			}
			for ib, t := range m {
				g := G[ib]
				if g == nil {
					G[ib] = &acc{dot: t.dot, n2a: t.n2a, n2b: t.n2b, c: t.c}
				} else {
					g.dot += t.dot
					g.n2a += t.n2a
					g.n2b += t.n2b
					g.c += t.c
				}
				pairsUpdated++
			}
		}
	}
	t1 := time.Now()

	// Top-K por ítem
	out := make(map[int][]kv)
	var simsKept, lines uint64
	for i, m := range global {
		cands := make([]kv, 0, len(m))
		for j, t := range m {
			if t.c < minCo || t.n2a == 0 || t.n2b == 0 {
				continue
			}
			sim := t.dot / (math.Sqrt(t.n2a) * math.Sqrt(t.n2b))
			if !math.IsNaN(sim) && !math.IsInf(sim, 0) {
				cands = append(cands, kv{j: j, s: sim})
			}
		}
		cands = topK(cands, k)
		out[i] = cands
		simsKept += uint64(len(cands))
	}

	// Escribir CSV
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

	rep = fmt.Sprintf(
		`== COSENO ITEM-BASED (concurrente) ==
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

// ==== MAIN (selector de modo) ====

func main() {
	var mode string
	var k, minCo int
	var pctUsers, pctItems int
	var workers int

	flag.StringVar(&mode, "mode", "item", "item | user")
	flag.IntVar(&k, "k", 20, "Top-K vecinos")
	flag.IntVar(&minCo, "min_co", 3, "mínimo co-ocurrencias para considerar similitud")
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
		rep, err = runItemBasedCosineConcurrent(k, minCo, pctUsers, pctItems, workers)
	case "user":
		rep, err = runUserBasedCosineConcurrent(k, minCo, pctUsers, pctItems, workers)
	default:
		panic("modo inválido: use --mode=item | --mode=user")
	}
	if err != nil {
		panic(err)
	}
	fmt.Print(rep)
}

// ==== util lectura binaria CSR ====

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
