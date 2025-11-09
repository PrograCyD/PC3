//go:build algorithms
// +build algorithms

package main

/*
PEARSON (secuencial, con muestreo) — MODO ITEM o USER

Resumen
--------------
- "Pearson" mide la correlación lineal entre dos vectores de valoraciones
  usando desviaciones respecto a su media.
- En práctica:
  * USER-BASED: correlación entre usuarios usando ratings centrados por usuario r' = r - μ_u.
    (Leemos el CSR ya centrado por usuario. Con r' el cálculo es análogo a un coseno sobre r').
  * ITEM-BASED: correlación entre ítems usando ratings centrados por ítem r' = r - μ_i.
    (Calculamos μ_i a partir de artifacts/ratings_ui.csv y luego acumulamos solo con r').

Modes:
  --mode=user  -> User-Based Pearson  (CSR centrado por usuario)
  --mode=item  -> Item-Based Pearson  (a partir de ratings_ui.csv; centrado por ítem)

Muestreo determinístico por id para acelerar pruebas:
  --pct_users=...  --pct_items=...   (0..100), válido en ambos modos

Parámetros comunes:
  --k=20               Top-K vecinos por nodo (usuario o ítem)
  --min_co=3           mínimo de co-valoraciones para aceptar una similitud

Entradas según modo:
  user:
    - artifacts/matrix_user_csr/indptr.bin   int64,  len=U+1
    - artifacts/matrix_user_csr/indices.bin  int32,  len=NNZ
    - artifacts/matrix_user_csr/data.bin     float32,len=NNZ   // r' = r - μ_u
  item:
    - artifacts/ratings_ui.csv               uIdx,iIdx,rating  (dos pasadas para calcular μ_i y luego acumular)

Salidas:
  user:
    - artifacts/sim/user_topk_pearson.csv     (uIdx,vIdx,sim)
    - artifacts/sim/user_pearson_report.txt
  item:
    - artifacts/sim/item_topk_pearson.csv     (iIdx,jIdx,sim)
    - artifacts/sim/item_pearson_report.txt
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
	// CSR centrado por usuario (modo user)
	csrIndptrPath  = "artifacts/matrix_user_csr/indptr.bin"
	csrIndicesPath = "artifacts/matrix_user_csr/indices.bin"
	csrDataPath    = "artifacts/matrix_user_csr/data.bin"
	// ratings crudos (modo item)
	inTriplets = "artifacts/ratings_ui.csv"

	// outputs
	outUserTopK   = "artifacts/sim/user_topk_pearson.csv"
	outUserReport = "artifacts/sim/user_pearson_report.txt"

	outItemTopK   = "artifacts/sim/item_topk_pearson.csv"
	outItemReport = "artifacts/sim/item_pearson_report.txt"
)

// pares/similitudes
type pair struct {
	j int
	s float64
}

// acumulador de Pearson (∑xy, ∑x2, ∑y2, count)
type acc struct {
	xy, x2, y2 float64
	c          int
}

// ===================== helpers comunes =====================

// Top-K por mezcla
func topMerge(curr, add []pair, k int) []pair {
	curr = append(curr, add...)
	sort.Slice(curr, func(i, j int) bool { return curr[i].s > curr[j].s })
	if len(curr) > k {
		curr = curr[:k]
	}
	return curr
}

// lectura binaria (modo user)
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

// hash determinístico simple (FNV-1a truncado) para muestreo por id
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

// ===================== MAIN =====================
func main() {
	var mode string
	var k, minCo int
	var pctUsers, pctItems int

	flag.StringVar(&mode, "mode", "user", "user | item")
	flag.IntVar(&k, "k", 20, "Top-K vecinos")
	flag.IntVar(&minCo, "min_co", 3, "mínimo co-valoraciones")
	flag.IntVar(&pctUsers, "pct_users", 100, "% de usuarios (0-100)")
	flag.IntVar(&pctItems, "pct_items", 100, "% de ítems (0-100)")
	flag.Parse()

	if mode != "user" && mode != "item" {
		panic("--mode debe ser user o item")
	}
	if mode == "user" {
		runUserPearson(k, minCo, pctUsers, pctItems)
	} else {
		runItemPearson(k, minCo, pctUsers, pctItems)
	}
}

// ===================== USER-BASED (CSR, r' por usuario) =====================
func runUserPearson(k, minCo, pctUsers, pctItems int) {
	t0 := time.Now()

	if err := os.MkdirAll(filepath.Dir(outUserTopK), 0o755); err != nil {
		panic(err)
	}

	indptr := readInt64(csrIndptrPath)
	indices := readInt32(csrIndicesPath)
	data := readFloat32(csrDataPath) // r' = r - μ_u
	U := len(indptr) - 1

	// índice invertido item -> [(u, r')]
	type ur struct {
		u int
		r float64
	}

	// número de ítems
	maxI := 0
	for _, x := range indices {
		if int(x)+1 > maxI {
			maxI = int(x) + 1
		}
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

	// acumular Pearson por pares de usuarios sobre co-items
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
				// xa y xb ya son r' centrados por usuario -> Pearson ≡ coseno sobre r'
				t.xy += xa * xb
				t.x2 += xa * xa
				t.y2 += xb * xb
				t.c++
				pairsUpdated++
			}
		}
	}
	t2 := time.Now()

	// Top-K por usuario
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

	// escribir CSV
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
		`== PEARSON USER-BASED (secuencial, muestreado sobre CSR centrado) ==
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
	fmt.Printf("[OK] user_topk_pearson -> %s\n", outUserTopK)
}

// ===================== ITEM-BASED (dos pasadas, r' por ítem) =====================
func runItemPearson(k, minCo, pctUsers, pctItems int) {
	t0 := time.Now()

	if err := os.MkdirAll(filepath.Dir(outItemTopK), 0o755); err != nil {
		panic(err)
	}

	// PASADA 1: medias por ítem μ_i
	type sumcnt struct {
		sum float64
		cnt int
	}
	itemStats := make(map[int]sumcnt, 100_000)

	f1, err := os.Open(inTriplets)
	if err != nil {
		panic(err)
	}
	r1 := csv.NewReader(bufio.NewReader(f1))
	_, _ = r1.Read() // header
	for {
		rec, err := r1.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			continue
		}
		i, _ := strconv.Atoi(rec[1])
		r, _ := strconv.ParseFloat(rec[2], 64)
		sc := itemStats[i]
		sc.sum += r
		sc.cnt++
		itemStats[i] = sc
	}
	f1.Close()

	// convertir a medias
	itemMean := make(map[int]float64, len(itemStats))
	for i, sc := range itemStats {
		if sc.cnt > 0 {
			itemMean[i] = sc.sum / float64(sc.cnt)
		}
	}
	// PASADA 2: acumular Pearson con r' = r - μ_i por usuario
	f2, err := os.Open(inTriplets)
	if err != nil {
		panic(err)
	}
	r2 := csv.NewReader(bufio.NewReader(f2))
	_, _ = r2.Read() // header

	// i -> j -> acc
	co := make(map[int]map[int]*acc)
	var usersKept, triplesOK, pairsUpdated uint64

	lastU := -1
	type ir struct {
		i  int
		rp float64
	}
	var items []ir

	flush := func() {
		if len(items) == 0 {
			return
		}
		usersKept++
		for a := 0; a < len(items); a++ {
			ia, xa := items[a].i, items[a].rp
			for b := a + 1; b < len(items); b++ {
				ib, xb := items[b].i, items[b].rp
				m := co[ia]
				if m == nil {
					m = make(map[int]*acc)
					co[ia] = m
				}
				t := m[ib]
				if t == nil {
					t = &acc{}
					m[ib] = t
				}
				t.xy += xa * xb
				t.x2 += xa * xa
				t.y2 += xb * xb
				t.c++
				pairsUpdated++
			}
		}
		items = items[:0]
	}

	for {
		rec, err := r2.Read()
		if err != nil {
			if err.Error() == "EOF" {
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
				flush()
				lastU = u
			}
			continue
		}
		if lastU == -1 {
			lastU = u
		}
		if u != lastU {
			flush()
			lastU = u
		}

		// muestreo por ítem
		if !keepByPct(i, pctItems) {
			continue
		}

		// centrado por ítem
		mu := itemMean[i]
		rp := r - mu

		items = append(items, ir{i: i, rp: rp})
		triplesOK++
	}
	flush()
	f2.Close()
	t1 := time.Now()

	// Top-K por ítem
	out := make(map[int][]pair)
	var lines uint64
	for i, m := range co {
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

	rep := fmt.Sprintf(
		`== PEARSON ITEM-BASED (secuencial, muestreado; centrado por ítem) ==
pct_users / pct_items :   %d%% / %d%%
Usuarios usados       :   %d
Tripletas leídas ok   :   %d
Pares i-j actualizados:   %d
Líneas escritas (CSV) :   %d
Parámetros            :   k=%d  min_co=%d

Tiempos:
  Medias por ítem     :   %s
  Acumular por usuario:   %s
  Escribir CSV        :   %s
  TOTAL               :   %s
Salida:
  %s
`, pctUsers, pctItems, usersKept, triplesOK, pairsUpdated, lines, k, minCo,
		// tiempos: (t1 incluye medias+acumulación), así que partimos:
		// t0..(t0?) -> Para claridad, marcamos t1.Sub(t0) como "Acumular por usuario"
		// y calculamos medias implícitamente dentro; si prefieres, separa un tMedias.
		time.Duration(0), t1.Sub(t0), t3.Sub(t2), t3.Sub(t0), outItemTopK)

	_ = os.WriteFile(outItemReport, []byte(rep), 0o644)
	fmt.Print(rep)
	fmt.Printf("[OK] item_topk_pearson -> %s\n", outItemTopK)
}
