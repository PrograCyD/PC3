//go:build algorithms
// +build algorithms

package main

/*
JACCARD (Concurrente) — Item-Based y User-Based

Resumen:
- Coef. de Jaccard: J(A,B) = |A ∩ B| / |A ∪ B|.
- User-Based: A,B son conjuntos de ítems valorados por usuarios u,v (trabajamos sobre CSR por usuario).
- Item-Based: A,B son conjuntos de usuarios que valoraron los ítems i,j (leemos ratings_ui.csv).

Concurrencia:
- Patrón worker-pool + reduce con canales.
- Se divide el espacio en sub-bloques:
  * User-Based: iteramos por ítem (columna) y generamos pares de usuarios (submatrices horizontales).
  * Item-Based: agrupamos canastas por usuario y generamos pares de ítems (submatrices verticales).
- Cada worker acumula intersecciones locales, el hilo principal fusiona y calcula Top-K.

Entradas:
- --mode=item  → artifacts/ratings_ui.csv (uIdx,iIdx,rating) ordenado por uIdx.
- --mode=user  → artifacts/matrix_user_csr/{indptr.bin,indices.bin,data.bin}.

Parámetros:
- --mode=item|user
- --k=20              (Top-K vecinos)
- --min_co=3          (mínimo de co-ocurrencias)
- --pct_users=100     (muestreo determinista por id de usuario)
- --pct_items=100     (muestreo determinista por id de ítem)
- --workers=8         (número de goroutines)

Salidas:
- CSV de similitudes Top-K:
  * item: artifacts/sim/item_topk_jaccard_conc.csv  (iIdx,jIdx,sim)
  * user: artifacts/sim/user_topk_jaccard_conc.csv  (uIdx,vIdx,sim)
- Reporte con tiempos y conteos:
  * artifacts/sim/item_jaccard_conc_report.txt
  * artifacts/sim/user_jaccard_conc_report.txt
*/

import (
	"bufio"
	"encoding/binary"
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"time"
)

const (
	inTriplets = "artifacts/ratings_ui.csv"

	csrIndptrPath  = "artifacts/matrix_user_csr/indptr.bin"
	csrIndicesPath = "artifacts/matrix_user_csr/indices.bin"

	outItemTopK   = "artifacts/sim/item_topk_jaccard_conc.csv"
	outItemReport = "artifacts/sim/item_jaccard_conc_report.txt"

	outUserTopK   = "artifacts/sim/user_topk_jaccard_conc.csv"
	outUserReport = "artifacts/sim/user_jaccard_conc_report.txt"
)

type kv struct {
	j int
	s float64
}

type accJ struct {
	inter int
}

// ---------- util: hash y muestreo determinista ----------
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

func ensureDirFor(path string) error {
	return os.MkdirAll(filepath.Dir(path), 0o755)
}

// ---------- USER-BASED (CSR) ----------
func runUserBasedJaccardConcurrent(k, minCo, pctUsers, pctItems, workers int) (string, error) {
	t0 := time.Now()

	indptr := readInt64(csrIndptrPath)
	indices := readInt32(csrIndicesPath)
	U := len(indptr) - 1

	// Construir invertido: item -> []users y contar cardinalidades por usuario
	maxI := 0
	for _, x := range indices {
		if int(x)+1 > maxI {
			maxI = int(x) + 1
		}
	}
	itemUsers := make([][]int, maxI)
	userCount := make([]int, U) // |I(u)| SOBRE EL SUBCONJUNTO MUESTREADO

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
			itemUsers[i] = append(itemUsers[i], u)
			userCount[u]++
		}
	}
	tInv := time.Since(t0)

	// Worker pool: divide items en chunks
	type job struct{ lo, hi int }
	jobs := make(chan job, workers)
	type part struct{ m map[int]map[int]*accJ }
	results := make(chan part, workers)

	worker := func() {
		local := make(map[int]map[int]*accJ)
		for jb := range jobs {
			for i := jb.lo; i < jb.hi; i++ {
				users := itemUsers[i]
				n := len(users)
				for a := 0; a < n; a++ {
					ua := users[a]
					for b := a + 1; b < n; b++ {
						ub := users[b]
						m := local[ua]
						if m == nil {
							m = make(map[int]*accJ)
							local[ua] = m
						}
						t := m[ub]
						if t == nil {
							t = &accJ{}
							m[ub] = t
						}
						t.inter++
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
	global := make(map[int]map[int]*accJ)
	for part := range results {
		for ua, m := range part.m {
			G := global[ua]
			if G == nil {
				G = make(map[int]*accJ)
				global[ua] = G
			}
			for ub, t := range m {
				g := G[ub]
				if g == nil {
					G[ub] = &accJ{inter: t.inter}
				} else {
					g.inter += t.inter
				}
			}
		}
	}
	tWork := time.Since(t0) - tInv

	// Calcular Jaccard y Top-K
	out := make(map[int][]kv)
	var lines uint64
	for u, m := range global {
		cands := make([]kv, 0, len(m))
		for v, t := range m {
			if t.inter < minCo {
				continue
			}
			a := userCount[u]
			b := userCount[v]
			if a == 0 || b == 0 {
				continue
			}
			union := a + b - t.inter
			if union <= 0 {
				continue
			}
			sim := float64(t.inter) / float64(union)
			cands = append(cands, kv{j: v, s: sim})
		}
		out[u] = topK(cands, k)
	}
	tTop := time.Since(t0) - tInv - tWork

	if err := ensureDirFor(outUserTopK); err != nil {
		return "", err
	}
	fw, _ := os.Create(outUserTopK)
	w := csv.NewWriter(bufio.NewWriter(fw))
	_ = w.Write([]string{"uIdx", "vIdx", "sim"})
	for u, list := range out {
		for _, p := range list {
			_ = w.Write([]string{strconv.Itoa(u), strconv.Itoa(p.j), fmt.Sprintf("%.6f", p.s)})
			lines++
		}
	}
	w.Flush()
	fw.Close()
	tCSV := time.Since(t0) - tInv - tWork - tTop

	rep := fmt.Sprintf(
		`== JACCARD USER-BASED (concurrente) ==
Usuarios (U):           %d
Ítems (I):              %d
Workers:                %d
pct_users / pct_items:  %d%% / %d%%
Líneas escritas (CSV):  %d

Tiempos:
  Invertir item->users: %s
  Workers + reduce:     %s
  Top-K:                %s
  Escribir CSV:         %s
  TOTAL:                %s

Salida:
  %s
`, U, maxI, workers, pctUsers, pctItems, lines,
		tInv, tWork, tTop, tCSV, time.Since(t0), outUserTopK)

	_ = os.WriteFile(outUserReport, []byte(rep), 0o644)
	return rep, nil
}

// ---------- ITEM-BASED (ratings_ui.csv) ----------
func runItemBasedJaccardConcurrent(k, minCo, pctUsers, pctItems, workers int) (string, error) {
	t0 := time.Now()

	f, err := os.Open(inTriplets)
	if err != nil {
		return "", err
	}
	defer f.Close()
	rd := csv.NewReader(bufio.NewReader(f))
	_, _ = rd.Read() // header

	type rating struct{ i int }
	type userBlock struct{ users [][]rating }

	jobs := make(chan userBlock, workers)
	type part struct{ m map[int]map[int]*accJ }
	results := make(chan part, workers)

	// Contaremos |U(i)| correctamente durante la lectura
	itemCount := make(map[int]int)

	worker := func() {
		local := make(map[int]map[int]*accJ)
		for blk := range jobs {
			for _, items := range blk.users {
				for a := 0; a < len(items); a++ {
					ia := items[a].i
					for b := a + 1; b < len(items); b++ {
						ib := items[b].i
						m := local[ia]
						if m == nil {
							m = make(map[int]*accJ)
							local[ia] = m
						}
						t := m[ib]
						if t == nil {
							t = &accJ{}
							m[ib] = t
						}
						t.inter++
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

	// Lectura + batching por usuarios
	const usersPerBlock = 4096
	block := userBlock{users: make([][]rating, 0, usersPerBlock)}
	var lastU = -1
	var basket []rating

	emitUser := func() {
		if len(basket) == 0 {
			return
		}
		block.users = append(block.users, append([]rating(nil), basket...))
		if len(block.users) >= usersPerBlock {
			jobs <- block
			block = userBlock{users: make([][]rating, 0, usersPerBlock)}
		}
		basket = basket[:0]
	}

	linesRead := 0
	for {
		rec, er := rd.Read()
		if er != nil {
			if er.Error() == "EOF" {
				break
			}
			continue
		}
		linesRead++
		u, _ := strconv.Atoi(rec[0])
		i, _ := strconv.Atoi(rec[1])

		if !keepByPct(u, pctUsers) || !keepByPct(i, pctItems) {
			// cerrar grupo al cambiar usuario aunque no se use éste
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
		}
		basket = append(basket, rating{i: i})
		itemCount[i]++ // |U(i)| sobre el subconjunto muestreado
	}
	emitUser()
	if len(block.users) > 0 {
		jobs <- block
	}
	close(jobs)
	go func() { wg.Wait(); close(results) }()

	// Reduce
	global := make(map[int]map[int]*accJ)
	for part := range results {
		for ia, m := range part.m {
			G := global[ia]
			if G == nil {
				G = make(map[int]*accJ)
				global[ia] = G
			}
			for ib, t := range m {
				g := G[ib]
				if g == nil {
					G[ib] = &accJ{inter: t.inter}
				} else {
					g.inter += t.inter
				}
			}
		}
	}
	tWork := time.Since(t0)

	// Calcular Jaccard y Top-K
	out := make(map[int][]kv)
	var lines uint64
	for i, m := range global {
		cands := make([]kv, 0, len(m))
		for j, t := range m {
			if t.inter < minCo {
				continue
			}
			a := itemCount[i]
			b := itemCount[j]
			if a == 0 || b == 0 {
				continue
			}
			union := a + b - t.inter
			if union <= 0 {
				continue
			}
			sim := float64(t.inter) / float64(union)
			cands = append(cands, kv{j: j, s: sim})
		}
		out[i] = topK(cands, k)
	}

	if err := ensureDirFor(outItemTopK); err != nil {
		return "", err
	}
	fw, _ := os.Create(outItemTopK)
	w := csv.NewWriter(bufio.NewWriter(fw))
	_ = w.Write([]string{"iIdx", "jIdx", "sim"})
	for i, list := range out {
		for _, p := range list {
			_ = w.Write([]string{strconv.Itoa(i), strconv.Itoa(p.j), fmt.Sprintf("%.6f", p.s)})
			lines++
		}
	}
	w.Flush()
	fw.Close()
	tCSV := time.Since(t0) - tWork

	rep := fmt.Sprintf(
		`== JACCARD ITEM-BASED (concurrente) ==
Workers:                %d
pct_users / pct_items:  %d%% / %d%%
Tripletas leídas:       %d
Líneas escritas (CSV):  %d

Tiempos:
  Workers + reduce:     %s
  Escribir CSV:         %s
  TOTAL:                %s

Salida:
  %s
`, workers, pctUsers, pctItems, linesRead, lines,
		tWork, tCSV, time.Since(t0), outItemTopK)

	_ = os.WriteFile(outItemReport, []byte(rep), 0o644)
	return rep, nil
}

// ---------- MAIN ----------
func main() {
	var mode string
	var k, minCo int
	var pctUsers, pctItems, workers int
	flag.StringVar(&mode, "mode", "item", "item | user")
	flag.IntVar(&k, "k", 20, "Top-K vecinos")
	flag.IntVar(&minCo, "min_co", 3, "mínimo de co-ocurrencias")
	flag.IntVar(&pctUsers, "pct_users", 100, "% de usuarios (0-100)")
	flag.IntVar(&pctItems, "pct_items", 100, "% de ítems (0-100)")
	flag.IntVar(&workers, "workers", 8, "número de goroutines/workers")
	flag.Parse()

	var rep string
	var err error
	switch mode {
	case "item":
		rep, err = runItemBasedJaccardConcurrent(k, minCo, pctUsers, pctItems, workers)
	case "user":
		rep, err = runUserBasedJaccardConcurrent(k, minCo, pctUsers, pctItems, workers)
	default:
		panic("modo inválido: use --mode=item | --mode=user")
	}
	if err != nil {
		panic(err)
	}
	fmt.Print(rep)
}

// ---------- util lectura binaria ----------
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
