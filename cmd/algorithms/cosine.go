//go:build algorithms
// +build algorithms

package main

/*
COSENO (secuencial, con muestreo) — MODO ITEM o USER

Modes:
  --mode=item  -> Item-Based Cosine  (usa artifacts/ratings_ui.csv; ratings crudos)
  --mode=user  -> User-Based Cosine  (usa artifacts/matrix_user_csr/*; ratings centrados r' = r - mean(u))

Muestreo determinístico por id (hash-based) para acelerar pruebas:
  --pct_users=...  --pct_items=...   (0..100), válido en ambos modos

Parámetros comunes:
  --k=20               Top-K vecinos por nodo (ítem o usuario)
  --min_co=3           mínimo de co-valoraciones para aceptar una similitud

Entradas según modo:
  item:
    - artifacts/ratings_ui.csv             (uIdx,iIdx,rating)  [ordenado por uIdx]
  user:
    - artifacts/matrix_user_csr/indptr.bin   int64,  len=U+1
    - artifacts/matrix_user_csr/indices.bin  int32,  len=NNZ
    - artifacts/matrix_user_csr/data.bin     float32,len=NNZ   // r' = r - mean(u)

Salidas:
  item:
    - artifacts/sim/item_topk_cosine.csv     (iIdx,jIdx,sim)
    - artifacts/sim/item_cosine_report.txt
  user:
    - artifacts/sim/user_topk_cosine.csv     (uIdx,vIdx,sim)
    - artifacts/sim/user_cosine_report.txt
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
	"time"
)

// ---- rutas de entrada/salida ----
const (
	// item-based
	inTriplets    = "artifacts/ratings_ui.csv"
	outItemTopK   = "artifacts/sim/item_topk_cosine.csv"
	outItemReport = "artifacts/sim/item_cosine_report.txt"
	// user-based (CSR centrado por usuario)
	csrIndptrPath  = "artifacts/matrix_user_csr/indptr.bin"
	csrIndicesPath = "artifacts/matrix_user_csr/indices.bin"
	csrDataPath    = "artifacts/matrix_user_csr/data.bin"
	outUserTopK    = "artifacts/sim/user_topk_cosine.csv"
	outUserReport  = "artifacts/sim/user_cosine_report.txt"
)

type pair struct {
	j int
	s float64
}

type acc struct {
	xy, x2, y2 float64
	c          int
}

// hash determinístico simple (FNV-1a) para muestreo por id
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

// mezcla ordenada de Top-K vecinos
func topMerge(curr, add []pair, k int) []pair {
	curr = append(curr, add...)
	sort.Slice(curr, func(i, j int) bool { return curr[i].s > curr[j].s })
	if len(curr) > k {
		curr = curr[:k]
	}
	return curr
}

// ---- utilidades lectura binaria (modo user) ----
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

// ===================== MAIN =====================
func main() {
	var mode string
	var k, minCo int
	var pctUsers, pctItems int

	flag.StringVar(&mode, "mode", "item", "item | user")
	flag.IntVar(&k, "k", 20, "Top-K vecinos")
	flag.IntVar(&minCo, "min_co", 3, "mínimo co-valoraciones")
	flag.IntVar(&pctUsers, "pct_users", 100, "% de usuarios (0-100)")
	flag.IntVar(&pctItems, "pct_items", 10, "% de ítems (0-100)")
	flag.Parse()

	if mode != "item" && mode != "user" {
		panic("--mode debe ser item o user")
	}

	if mode == "item" {
		runItemCosine(k, minCo, pctUsers, pctItems)
	} else {
		runUserCosine(k, minCo, pctUsers, pctItems)
	}
}

// ===================== ITEM-BASED =====================
func runItemCosine(k, minCo, pctUsers, pctItems int) {
	t0 := time.Now()

	if err := os.MkdirAll(filepath.Dir(outItemTopK), 0o755); err != nil {
		panic(err)
	}

	f, err := os.Open(inTriplets)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	rd := csv.NewReader(bufio.NewReader(f))
	_, _ = rd.Read() // header

	// i -> j -> acumuladores
	dot := make(map[int]map[int]*acc)
	// buffers por usuario
	lastU := -1
	var items []pair // reuse as (j=iIdx, s=rating)

	var usersKept, triplesOK, pairsUpdated, lines uint64

	flush := func() {
		if len(items) == 0 {
			return
		}
		usersKept++
		for a := 0; a < len(items); a++ {
			ia, ra := items[a].j, items[a].s
			for b := a + 1; b < len(items); b++ {
				ib, rb := items[b].j, items[b].s
				m := dot[ia]
				if m == nil {
					m = make(map[int]*acc)
					dot[ia] = m
				}
				t := m[ib]
				if t == nil {
					t = &acc{}
					m[ib] = t
				}
				t.xy += ra * rb
				t.x2 += ra * ra
				t.y2 += rb * rb
				t.c++
				pairsUpdated++
			}
		}
		items = items[:0]
	}

	for {
		rec, err := rd.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			continue
		}
		u, _ := strconv.Atoi(rec[0])
		i, _ := strconv.Atoi(rec[1])
		r, _ := strconv.ParseFloat(rec[2], 64)

		// muestreo usuario
		if !keepByPct(u, pctUsers) {
			if lastU != -1 && u != lastU {
				flush()
				lastU = u
			}
			continue
		}
		// cortar por cambio usuario
		if lastU == -1 {
			lastU = u
		}
		if u != lastU {
			flush()
			lastU = u
		}

		// muestreo item
		if !keepByPct(i, pctItems) {
			continue
		}

		items = append(items, pair{j: i, s: r})
		triplesOK++
	}
	flush()
	t1 := time.Now()

	// Top-K por ítem
	out := make(map[int][]pair)
	for i, m := range dot {
		cands := make([]pair, 0, len(m))
		for j, t := range m {
			if t.c < minCo || t.x2 == 0 || t.y2 == 0 {
				continue
			}
			sim := t.xy / (math.Sqrt(t.x2) * math.Sqrt(t.y2))
			if !math.IsNaN(sim) && !math.IsInf(sim, 0) {
				cands = append(cands, pair{j: j, s: sim})
			}
		}
		sort.Slice(cands, func(a, b int) bool { return cands[a].s > cands[b].s })
		if len(cands) > k {
			cands = cands[:k]
		}
		out[i] = cands
	}
	t2 := time.Now()

	// escribir CSV
	fw, _ := os.Create(outItemTopK)
	defer fw.Close()
	w := csv.NewWriter(bufio.NewWriter(fw))
	defer w.Flush()
	_ = w.Write([]string{"iIdx", "jIdx", "sim"})
	for i, list := range out {
		for _, p := range list {
			_ = w.Write([]string{strconv.Itoa(i), strconv.Itoa(p.j), fmt.Sprintf("%.6f", p.s)})
			lines++
		}
	}
	t3 := time.Now()

	// reporte
	rep := fmt.Sprintf(
		`== COSENO ITEM-BASED (secuencial, muestreado) ==
pct_users / pct_items :   %d%% / %d%%
Usuarios usados       :   %d
Tripletas leídas ok   :   %d
Pares i-j actualizados:   %d
Líneas escritas (CSV) :   %d
Parámetros            :   k=%d  min_co=%d

Tiempos:
  Acumular por usuario:   %s
  Top-K por ítem      :   %s
  Escribir CSV        :   %s
  TOTAL               :   %s
Salida:
  %s
`, pctUsers, pctItems, usersKept, triplesOK, pairsUpdated, lines, k, minCo,
		t1.Sub(t0), t2.Sub(t1), t3.Sub(t2), t3.Sub(t0), outItemTopK)
	_ = os.WriteFile(outItemReport, []byte(rep), 0o644)
	fmt.Print(rep)
	fmt.Printf("[OK] item_topk_cosine -> %s\n", outItemTopK)
}

// ===================== USER-BASED =====================
// Construye similitud Coseno entre usuarios utilizando CSR con r' (centrado).
func runUserCosine(k, minCo, pctUsers, pctItems int) {
	t0 := time.Now()

	if err := os.MkdirAll(filepath.Dir(outUserTopK), 0o755); err != nil {
		panic(err)
	}

	indptr := readInt64(csrIndptrPath)
	indices := readInt32(csrIndicesPath)
	data := readFloat32(csrDataPath)

	U := len(indptr) - 1

	// Para construir el índice invertido por ítem: i -> [(u, r')]
	// (aplicando muestreo por usuario y por ítem)
	// Primero, obtener número de ítems máximo:
	maxI := 0
	for _, x := range indices {
		if int(x)+1 > maxI {
			maxI = int(x) + 1
		}
	}
	type ur struct {
		u int
		r float64
	}
	itemUsers := make([][]ur, maxI)

	var triplesOK uint64
	for u := 0; u < U; u++ {
		if !keepByPct(u, pctUsers) {
			continue
		}
		for p := indptr[u]; p < indptr[u+1]; p++ {
			i := int(indices[p])
			if !keepByPct(i, pctItems) {
				continue
			}
			rp := float64(data[p])
			itemUsers[i] = append(itemUsers[i], ur{u, rp})
			triplesOK++
		}
	}
	t1 := time.Now()

	// Acumular coseno por pares (usuarios que co-valoraron un ítem)
	co := make(map[uint64]*acc, 8_000_000)
	var pairsUpdated uint64

	key := func(a, b int) uint64 {
		if a > b {
			a, b = b, a
		}
		return (uint64(a) << 32) | uint64(b)
	}

	for i := 0; i < maxI; i++ {
		users := itemUsers[i]
		n := len(users)
		for a := 0; a < n; a++ {
			ua, xa := users[a].u, users[a].r
			for b := a + 1; b < n; b++ {
				ub, xb := users[b].u, users[b].r
				kp := key(ua, ub)
				t := co[kp]
				if t == nil {
					t = &acc{}
					co[kp] = t
				}
				t.xy += xa * xb
				t.x2 += xa * xa
				t.y2 += xb * xb
				t.c++
				pairsUpdated++
			}
		}
	}
	t2 := time.Now()

	// Convertir a Top-K por usuario
	out := make([][]pair, U)
	var simsKept, lines uint64

	for kv, t := range co {
		if t.c < minCo || t.x2 == 0 || t.y2 == 0 {
			continue
		}
		sim := t.xy / (math.Sqrt(t.x2) * math.Sqrt(t.y2))
		if math.IsNaN(sim) || math.IsInf(sim, 0) {
			continue
		}
		u := int(kv >> 32)
		v := int(kv & 0xffffffff)
		out[u] = topMerge(out[u], []pair{{j: v, s: sim}}, k)
		out[v] = topMerge(out[v], []pair{{j: u, s: sim}}, k)
		simsKept++
	}
	t3 := time.Now()

	// Escribir CSV
	f, _ := os.Create(outUserTopK)
	defer f.Close()
	w := csv.NewWriter(bufio.NewWriter(f))
	defer w.Flush()
	_ = w.Write([]string{"uIdx", "vIdx", "sim"})
	for u := 0; u < U; u++ {
		for _, p := range out[u] {
			_ = w.Write([]string{fmt.Sprintf("%d", u), fmt.Sprintf("%d", p.j), fmt.Sprintf("%.6f", p.s)})
			lines++
		}
	}
	t4 := time.Now()

	rep := fmt.Sprintf(
		`== COSENO USER-BASED (secuencial, muestreado sobre CSR centrado) ==
pct_users / pct_items :   %d%% / %d%%
Usuarios totales (U)  :   %d
Tripletas usadas (r') :   %d
Pares u-v actualizados:   %d
Similitudes retenidas :   %d
Líneas escritas (CSV) :   %d
Parámetros            :   k=%d  min_co=%d

Tiempos:
  Cargar/Invertir CSR :   %s
  Acumular pares      :   %s
  Top-K por usuario   :   %s
  Escribir CSV        :   %s
  TOTAL               :   %s
Salida:
  %s
`, pctUsers, pctItems, U, triplesOK, pairsUpdated, simsKept, lines, k, minCo,
		t1.Sub(t0), t2.Sub(t1), t3.Sub(t2), t4.Sub(t3), t4.Sub(t0), outUserTopK)
	_ = os.WriteFile(outUserReport, []byte(rep), 0o644)
	fmt.Print(rep)
	fmt.Printf("[OK] user_topk_cosine -> %s\n", outUserTopK)
}
