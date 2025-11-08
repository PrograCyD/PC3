//go:build algorithms
// +build algorithms

package main

/*
PEARSON USER-BASED (secuencial)
Lee la matriz CSR centrada por usuario y construye la similitud Pearson entre usuarios
mediante acumulación por ítem (co-valoraciones). Devuelve Top-K vecinos por usuario.

Entradas:
  - artifacts/matrix_user_csr/indptr.bin   (int64, len=U+1)
  - artifacts/matrix_user_csr/indices.bin  (int32, len=NNZ)  // movie indices
  - artifacts/matrix_user_csr/data.bin     (float32, len=NNZ) // r' = r - mean_u
  - artifacts/user_means.csv               (uIdx,mean)        // para predicción luego (no para similitud)

Parámetros (flags opcionales):
  --k=20           Top-K vecinos por usuario
  --min_co=3       mínimo de co-valoraciones para considerar similitud

Salida:
  - artifacts/sim/user_topk_pearson.csv  (uIdx,vIdx,sim)
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
)

const (
	indptrPath  = "artifacts/matrix_user_csr/indptr.bin"
	indicesPath = "artifacts/matrix_user_csr/indices.bin"
	dataPath    = "artifacts/matrix_user_csr/data.bin"

	outUserTopK = "artifacts/sim/user_topk_pearson.csv"
)

type pair struct {
	v int
	s float64
}

func main() {
	var k int
	var minCo int
	flag.IntVar(&k, "k", 20, "Top-K vecinos por usuario")
	flag.IntVar(&minCo, "min_co", 3, "mínimo co-valoraciones")
	flag.Parse()

	if err := os.MkdirAll(filepath.Dir(outUserTopK), 0o755); err != nil {
		panic(err)
	}

	indptr := readInt64(indptrPath)
	indices := readInt32(indicesPath)
	data := readFloat32(dataPath)

	U := len(indptr) - 1

	// Construimos índice invertido por ítem: para cada item -> lista (user, r')
	// A partir del CSR de usuarios
	maxI := 0
	for _, x := range indices {
		if int(x)+1 > maxI {
			maxI = int(x) + 1
		}
	}
	itemUsers := make([][]pair, maxI) // pair{u, r'}
	for u := 0; u < U; u++ {
		start := indptr[u]
		end := indptr[u+1]
		for p := start; p < end; p++ {
			i := int(indices[p])
			rp := float64(data[p])
			itemUsers[i] = append(itemUsers[i], pair{v: u, s: rp})
		}
	}

	// Acumular Pearson por pares de usuarios que co-valoraron al menos un ítem
	// Usamos mapas por usuario destino para sumar (∑xy, ∑x2, ∑y2, count)
	type acc struct {
		xy, x2, y2 float64
		c          int
	}
	out := make([][]pair, U)

	for i := 0; i < maxI; i++ {
		users := itemUsers[i]
		n := len(users)
		for a := 0; a < n; a++ {
			ua, xa := users[a].v, users[a].s
			// acumuladores temporales para ua
			tmp := make(map[int]*acc, n)
			for b := a + 1; b < n; b++ {
				ub, xb := users[b].v, users[b].s
				t := tmp[ub]
				if t == nil {
					t = &acc{}
					tmp[ub] = t
				}
				t.xy += xa * xb
				t.x2 += xa * xa
				t.y2 += xb * xb
				t.c++
			}
			// volcar a estructura global del usuario ua
			// (para no usar memoria cuadrática, convertimos directo a topK)
			cands := make([]pair, 0, len(tmp))
			for v, t := range tmp {
				if t.c < minCo || t.x2 == 0 || t.y2 == 0 {
					continue
				}
				sim := t.xy / (math.Sqrt(t.x2) * math.Sqrt(t.y2))
				if !math.IsNaN(sim) && !math.IsInf(sim, 0) {
					cands = append(cands, pair{v: v, s: sim})
				}
			}
			if len(cands) > 0 {
				out[ua] = topMerge(out[ua], cands, k)
			}
		}
	}

	// Escribir Top-K
	f, _ := os.Create(outUserTopK)
	defer f.Close()
	w := csv.NewWriter(bufio.NewWriter(f))
	defer w.Flush()
	_ = w.Write([]string{"uIdx", "vIdx", "sim"})
	for u := 0; u < U; u++ {
		for _, p := range out[u] {
			_ = w.Write([]string{
				fmt.Sprintf("%d", u),
				fmt.Sprintf("%d", p.v),
				fmt.Sprintf("%.6f", p.s),
			})
		}
	}
	fmt.Printf("[OK] user_topk_pearson -> %s\n", outUserTopK)
}

func topMerge(curr, add []pair, k int) []pair {
	curr = append(curr, add...)
	sort.Slice(curr, func(i, j int) bool { return curr[i].s > curr[j].s })
	if len(curr) > k {
		curr = curr[:k]
	}
	return curr
}

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
