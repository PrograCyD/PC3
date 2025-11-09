//go:build algorithms
// +build algorithms

package main

/*
JACCARD (secuencial, con muestreo) — MODO ITEM o USER

Resumen
-------
La similitud de Jaccard entre dos conjuntos A y B es:
    J(A,B) = |A ∩ B| / |A ∪ B| = inter / (deg(A) + deg(B) - inter)

En recomendación colaborativa:

- USER-BASED (mode=user):
    * Cada usuario u es un conjunto de ítems calificados I(u).
    * J(u,v) = |I(u) ∩ I(v)| / |I(u) ∪ I(v)|
    * Construimos co-ocurrencias de pares (u,v) a partir de los usuarios que coinciden en algún ítem.

- ITEM-BASED (mode=item):
    * Cada ítem i es un conjunto de usuarios U(i) que lo calificaron.
    * J(i,j) = |U(i) ∩ U(j)| / |U(i) ∪ U(j)|
    * Construimos co-ocurrencias de pares (i,j) a partir de los ítems coincidentes por usuario.

Muestreo determinístico por id
------------------------------
--pct_users=.. y --pct_items=.. (0..100) para acelerar pruebas con cortes reproducibles.
Se decide por id (hash-based), no por fila, para evitar sesgos.

Entradas (ambos modos)
----------------------
- artifacts/ratings_ui.csv   (uIdx,iIdx,rating)  // solo se usa presencia (implícito 1)

Parámetros
----------
--mode=user|item    (User-Based o Item-Based)
--k=20              (Top-K vecinos)
--min_co=3          (mínimo intersecciones para aceptar similitud)
--pct_users=100     (porcentaje de usuarios a considerar)
--pct_items=100     (porcentaje de ítems a considerar)

Salidas
-------
mode=user:
  - artifacts/sim/user_topk_jaccard.csv   (uIdx,vIdx,sim)
  - artifacts/sim/user_jaccard_report.txt
mode=item:
  - artifacts/sim/item_topk_jaccard.csv   (iIdx,jIdx,sim)
  - artifacts/sim/item_jaccard_report.txt
*/

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"time"
)

// -------- rutas de IO ----------
const (
	inTriplets = "artifacts/ratings_ui.csv"

	outUserTopK   = "artifacts/sim/user_topk_jaccard.csv"
	outUserReport = "artifacts/sim/user_jaccard_report.txt"

	outItemTopK   = "artifacts/sim/item_topk_jaccard.csv"
	outItemReport = "artifacts/sim/item_jaccard_report.txt"
)

// -------- estructuras auxiliares ----------
type pair struct {
	j int
	s float64
}

type accInt struct {
	inter int // intersección (co-ocurrencias)
}

// -------- utilidades comunes ----------
func topMerge(curr, add []pair, k int) []pair {
	curr = append(curr, add...)
	sort.Slice(curr, func(i, j int) bool { return curr[i].s > curr[j].s })
	if len(curr) > k {
		curr = curr[:k]
	}
	return curr
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

func main() {
	var mode string
	var k, minCo int
	var pctUsers, pctItems int

	flag.StringVar(&mode, "mode", "item", "user | item")
	flag.IntVar(&k, "k", 20, "Top-K vecinos")
	flag.IntVar(&minCo, "min_co", 3, "mínimo co-valoraciones (intersecciones)")
	flag.IntVar(&pctUsers, "pct_users", 100, "% de usuarios (0-100)")
	flag.IntVar(&pctItems, "pct_items", 100, "% de ítems (0-100)")
	flag.Parse()

	if err := os.MkdirAll("artifacts/sim", 0o755); err != nil {
		panic(err)
	}

	switch mode {
	case "user":
		runUserJaccard(k, minCo, pctUsers, pctItems)
	case "item":
		runItemJaccard(k, minCo, pctUsers, pctItems)
	default:
		panic("--mode debe ser user o item")
	}
}

// ===================== USER-BASED =====================
// J(u,v) = |I(u)∩I(v)| / (deg[u] + deg[v] - |I(u)∩I(v)|)
func runUserJaccard(k, minCo, pctUsers, pctItems int) {
	t0 := time.Now()

	// 1) Construir invertido: item -> []users (muestreado)
	type rec struct{ u, i int }
	f, err := os.Open(inTriplets)
	if err != nil {
		panic(err)
	}
	rd := csv.NewReader(bufio.NewReader(f))
	_, _ = rd.Read() // header

	itemUsers := make(map[int][]int) // i -> usuarios
	userDeg := make(map[int]int)     // deg[u] = |I(u)|
	seenItems := make(map[int]struct{})
	seenUsers := make(map[int]struct{})

	var triplesOK uint64
	for {
		row, err := rd.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			continue
		}
		u, _ := strconv.Atoi(row[0])
		i, _ := strconv.Atoi(row[1])
		// r := row[2] // no usado, binario

		if !keepByPct(u, pctUsers) || !keepByPct(i, pctItems) {
			continue
		}
		itemUsers[i] = append(itemUsers[i], u)
		userDeg[u]++
		seenUsers[u] = struct{}{}
		seenItems[i] = struct{}{}
		triplesOK++
	}
	f.Close()
	t1 := time.Now()

	// 2) Acumular intersecciones por pares de usuarios
	co := make(map[uint64]*accInt, 8_000_000)
	var pairsUpdated uint64
	key := func(a, b int) uint64 {
		if a > b {
			a, b = b, a
		}
		return (uint64(a) << 32) | uint64(b)
	}

	for _, users := range itemUsers {
		n := len(users)
		for a := 0; a < n; a++ {
			ua := users[a]
			for b := a + 1; b < n; b++ {
				ub := users[b]
				kp := key(ua, ub)
				t := co[kp]
				if t == nil {
					t = &accInt{}
					co[kp] = t
				}
				t.inter++
				pairsUpdated++
			}
		}
	}
	t2 := time.Now()

	// 3) Top-K por usuario
	out := make(map[int][]pair)
	var simsKept, lines uint64
	for kv, t := range co {
		if t.inter < minCo {
			continue
		}
		u := int(kv >> 32)
		v := int(kv & 0xffffffff)
		du := userDeg[u]
		dv := userDeg[v]
		if du == 0 || dv == 0 {
			continue
		}
		union := du + dv - t.inter
		if union <= 0 {
			continue
		}
		sim := float64(t.inter) / float64(union)
		if math.IsNaN(sim) || math.IsInf(sim, 0) {
			continue
		}
		out[u] = topMerge(out[u], []pair{{j: v, s: sim}}, k)
		out[v] = topMerge(out[v], []pair{{j: u, s: sim}}, k)
		simsKept++
	}

	// 4) Escribir CSV
	fw, _ := os.Create(outUserTopK)
	defer fw.Close()
	w := csv.NewWriter(bufio.NewWriter(fw))
	defer w.Flush()
	_ = w.Write([]string{"uIdx", "vIdx", "sim"})
	for u, lst := range out {
		for _, p := range lst {
			_ = w.Write([]string{strconv.Itoa(u), strconv.Itoa(p.j), fmt.Sprintf("%.6f", p.s)})
			lines++
		}
	}
	t3 := time.Now()

	// 5) Reporte
	rep := fmt.Sprintf(
		`== JACCARD USER-BASED (secuencial, muestreado) ==
pct_users / pct_items :   %d%% / %d%%
Usuarios usados       :   %d
Items usados          :   %d
Tripletas leídas ok   :   %d
Pares u-v actualizados:   %d
Similitudes retenidas :   %d
Líneas escritas (CSV) :   %d
Parámetros            :   k=%d  min_co=%d

Tiempos:
  Construir invertido :   %s
  Acumular intersecciones: %s
  Escribir CSV        :   %s
  TOTAL               :   %s
Salida:
  %s
`, pctUsers, pctItems, len(seenUsers), len(seenItems), triplesOK, pairsUpdated, simsKept, lines, k, minCo,
		t1.Sub(t0), t2.Sub(t1), t3.Sub(t2), t3.Sub(t0), outUserTopK)
	_ = os.WriteFile(outUserReport, []byte(rep), 0o644)
	fmt.Print(rep)
	fmt.Printf("[OK] user_topk_jaccard -> %s\n", outUserTopK)
}

// ===================== ITEM-BASED =====================
// J(i,j) = |U(i)∩U(j)| / (deg[i] + deg[j] - |U(i)∩U(j)|)
func runItemJaccard(k, minCo, pctUsers, pctItems int) {
	t0 := time.Now()

	// 1) Construir por usuario: u -> []items (muestreado)
	f, err := os.Open(inTriplets)
	if err != nil {
		panic(err)
	}
	rd := csv.NewReader(bufio.NewReader(f))
	_, _ = rd.Read() // header

	userItems := make(map[int][]int) // u -> items
	itemDeg := make(map[int]int)     // deg[i] = |U(i)|
	seenUsers := make(map[int]struct{})
	seenItems := make(map[int]struct{})
	var triplesOK uint64

	for {
		row, err := rd.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			continue
		}
		u, _ := strconv.Atoi(row[0])
		i, _ := strconv.Atoi(row[1])

		if !keepByPct(u, pctUsers) || !keepByPct(i, pctItems) {
			continue
		}
		userItems[u] = append(userItems[u], i)
		itemDeg[i]++
		seenUsers[u] = struct{}{}
		seenItems[i] = struct{}{}
		triplesOK++
	}
	f.Close()
	t1 := time.Now()

	// 2) Acumular intersecciones por pares de ítems dentro de cada usuario
	co := make(map[uint64]*accInt, 8_000_000)
	var pairsUpdated uint64
	key := func(a, b int) uint64 {
		if a > b {
			a, b = b, a
		}
		return (uint64(a) << 32) | uint64(b)
	}

	for _, items := range userItems {
		n := len(items)
		for a := 0; a < n; a++ {
			ia := items[a]
			for b := a + 1; b < n; b++ {
				ib := items[b]
				kp := key(ia, ib)
				t := co[kp]
				if t == nil {
					t = &accInt{}
					co[kp] = t
				}
				t.inter++
				pairsUpdated++
			}
		}
	}
	t2 := time.Now()

	// 3) Top-K por ítem
	out := make(map[int][]pair)
	var simsKept, lines uint64
	for kv, t := range co {
		if t.inter < minCo {
			continue
		}
		i := int(kv >> 32)
		j := int(kv & 0xffffffff)
		di := itemDeg[i]
		dj := itemDeg[j]
		if di == 0 || dj == 0 {
			continue
		}
		union := di + dj - t.inter
		if union <= 0 {
			continue
		}
		sim := float64(t.inter) / float64(union)
		if math.IsNaN(sim) || math.IsInf(sim, 0) {
			continue
		}
		out[i] = topMerge(out[i], []pair{{j: j, s: sim}}, k)
		out[j] = topMerge(out[j], []pair{{j: i, s: sim}}, k)
		simsKept++
	}

	// 4) Escribir CSV
	fw, _ := os.Create(outItemTopK)
	defer fw.Close()
	w := csv.NewWriter(bufio.NewWriter(fw))
	defer w.Flush()
	_ = w.Write([]string{"iIdx", "jIdx", "sim"})
	for i, lst := range out {
		for _, p := range lst {
			_ = w.Write([]string{strconv.Itoa(i), strconv.Itoa(p.j), fmt.Sprintf("%.6f", p.s)})
			lines++
		}
	}
	t3 := time.Now()

	// 5) Reporte
	rep := fmt.Sprintf(
		`== JACCARD ITEM-BASED (secuencial, muestreado) ==
pct_users / pct_items :   %d%% / %d%%
Usuarios usados       :   %d
Items usados          :   %d
Tripletas leídas ok   :   %d
Pares i-j actualizados:   %d
Similitudes retenidas :   %d
Líneas escritas (CSV) :   %d
Parámetros            :   k=%d  min_co=%d

Tiempos:
  Construir por usuario:   %s
  Acumular intersecciones: %s
  Escribir CSV          :   %s
  TOTAL                 :   %s
Salida:
  %s
`, pctUsers, pctItems, len(seenUsers), len(seenItems), triplesOK, pairsUpdated, simsKept, lines, k, minCo,
		t1.Sub(t0), t2.Sub(t1), t3.Sub(t2), t3.Sub(t0), outItemTopK)
	_ = os.WriteFile(outItemReport, []byte(rep), 0o644)
	fmt.Print(rep)
	fmt.Printf("[OK] item_topk_jaccard -> %s\n", outItemTopK)
}
