//go:build algorithms
// +build algorithms

package main

/*
COSENO ITEM-BASED (secuencial)
Acumula productos entre pares de ítems que aparecen en el mismo usuario.
Usa ratings crudos de artifacts/ratings_ui.csv (no centrados).

Entrada:
  - artifacts/ratings_ui.csv   (uIdx,iIdx,rating) ordenado por uIdx

Parámetros:
  --k=20
  --min_co=3

Salida:
  - artifacts/sim/item_topk_cosine.csv  (iIdx,jIdx,sim)
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
)

const inTriplets = "artifacts/ratings_ui.csv"
const outItemTopK = "artifacts/sim/item_topk_cosine.csv"

type kv struct {
	j int
	s float64
}

func main() {
	var k int
	var minCo int
	flag.IntVar(&k, "k", 20, "Top-K vecinos por ítem")
	flag.IntVar(&minCo, "min_co", 3, "mínimo co-valoraciones")
	flag.Parse()

	if err := os.MkdirAll(filepath.Dir(outItemTopK), 0o755); err != nil {
		panic(err)
	}

	// Leemos por usuario y acumulamos para pares de ítems (i,j)
	type acc struct {
		dot, n2i, n2j float64
		c             int
	}
	dot := make(map[int]map[int]*acc) // i -> j -> acc
	norm := make(map[int]float64)     // ||i||^2 (por si falta)

	f, err := os.Open(inTriplets)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	rd := csv.NewReader(bufio.NewReader(f))
	_, _ = rd.Read() // header

	// cargamos items de cada usuario en memoria temporal y procesamos pares
	var currU, lastU int = -1, -1
	var items []kv // (i, r)

	flush := func() {
		// acumular pares dentro del usuario
		for a := 0; a < len(items); a++ {
			ia, ra := items[a].j, items[a].s
			norm[ia] += ra * ra
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
				t.dot += ra * rb
				t.n2i += ra * ra
				t.n2j += rb * rb
				t.c++
			}
		}
		items = items[:0]
	}

	line := 0
	for {
		rec, err := rd.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			continue
		}
		line++
		u, _ := strconv.Atoi(rec[0])
		i, _ := strconv.Atoi(rec[1])
		r, _ := strconv.ParseFloat(rec[2], 64)

		if lastU == -1 {
			lastU = u
		}
		if u != lastU {
			flush()
			lastU = u
		}
		items = append(items, kv{j: i, s: r})
	}
	flush()

	// Top-K por ítem
	out := make(map[int][]kv)
	for i, m := range dot {
		cands := make([]kv, 0, len(m))
		for j, t := range m {
			if t.c < minCo || t.n2i == 0 || t.n2j == 0 {
				continue
			}
			sim := t.dot / (math.Sqrt(t.n2i) * math.Sqrt(t.n2j))
			if !math.IsNaN(sim) && !math.IsInf(sim, 0) {
				cands = append(cands, kv{j: j, s: sim})
			}
		}
		sort.Slice(cands, func(a, b int) bool { return cands[a].s > cands[b].s })
		if len(cands) > k {
			cands = cands[:k]
		}
		out[i] = cands
	}

	// Escribir
	fw, _ := os.Create(outItemTopK)
	defer fw.Close()
	w := csv.NewWriter(bufio.NewWriter(fw))
	defer w.Flush()
	_ = w.Write([]string{"iIdx", "jIdx", "sim"})
	for i, list := range out {
		for _, p := range list {
			_ = w.Write([]string{
				strconv.Itoa(i),
				strconv.Itoa(p.j),
				fmt.Sprintf("%.6f", p.s),
			})
		}
	}
	fmt.Printf("[OK] item_topk_cosine -> %s\n", outItemTopK)
}
