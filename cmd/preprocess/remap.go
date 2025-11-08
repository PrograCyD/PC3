//go:build remap
// +build remap

package main

/*
REMAPPING (userId→uIdx, movieId→iIdx) + TRIPLETS (uIdx,iIdx,rating)

Entrada:
  - artifacts/ratings_min5.csv  // resultado del filtrado (≥5 ratings por ítem)

Salidas:
  - artifacts/index/user_map.csv   (userId,uIdx)
  - artifacts/index/item_map.csv   (movieId,iIdx)
  - artifacts/ratings_ui.csv       (uIdx,iIdx,rating)  // ordenado por uIdx
  - artifacts/remap_report.txt     // resumen (U, I, NNZ)
*/

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type Triplet struct {
	U int
	I int
	R float64
}

const (
	inFiltered  = "artifacts/ratings_min5.csv"
	outTriplets = "artifacts/ratings_ui.csv"
	userMapPath = "artifacts/index/user_map.csv"
	itemMapPath = "artifacts/index/item_map.csv"
	remapReport = "artifacts/remap_report.txt"
)

func main() {
	if err := os.MkdirAll("artifacts/index", 0o755); err != nil {
		fmt.Printf("ERROR creando artifacts/index: %v\n", err)
		return
	}

	// 1) Primera pasada: construir mapas userId→uIdx, movieId→iIdx
	userIdx := make(map[int]int, 200000)
	itemIdx := make(map[int]int, 80000)
	var nextU, nextI int

	f, err := os.Open(inFiltered)
	if err != nil {
		fmt.Printf("ERROR abriendo %s: %v\n", inFiltered, err)
		return
	}
	reader := csv.NewReader(bufio.NewReader(f))
	reader.FieldsPerRecord = -1
	_, _ = reader.Read() // header

	buf := make([]Triplet, 0, 1_000_000)

	var nnz int64
	for {
		row, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			continue
		}
		if len(row) < 3 {
			continue
		}

		uid, err1 := strconv.Atoi(strings.TrimSpace(row[0]))
		iid, err2 := strconv.Atoi(strings.TrimSpace(row[1]))
		r, err3 := strconv.ParseFloat(strings.TrimSpace(row[2]), 64)
		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}

		u, ok := userIdx[uid]
		if !ok {
			u = nextU
			userIdx[uid] = u
			nextU++
		}
		i, ok := itemIdx[iid]
		if !ok {
			i = nextI
			itemIdx[iid] = i
			nextI++
		}

		buf = append(buf, Triplet{U: u, I: i, R: r})
		nnz++
	}
	f.Close()

	// 2) Ordenar por uIdx para facilitar CSR en el siguiente paso
	sort.Slice(buf, func(a, b int) bool {
		if buf[a].U == buf[b].U {
			return buf[a].I < buf[b].I
		}
		return buf[a].U < buf[b].U
	})

	// 3) Escribir triplets (uIdx,iIdx,rating)
	if err := writeTripletsCSV(outTriplets, buf); err != nil {
		fmt.Printf("ERROR escribiendo %s: %v\n", outTriplets, err)
		return
	}

	// 4) Escribir mapas
	if err := writeUserMap(userMapPath, userIdx); err != nil {
		fmt.Printf("ERROR escribiendo %s: %v\n", userMapPath, err)
		return
	}
	if err := writeItemMap(itemMapPath, itemIdx); err != nil {
		fmt.Printf("ERROR escribiendo %s: %v\n", itemMapPath, err)
		return
	}

	// 5) Reporte
	rep := fmt.Sprintf(
		"== REMAP ==\nUsuarios (U): %d\nItems (I): %d\nRatings (NNZ): %d\nSalida triplets: %s\n",
		len(userIdx), len(itemIdx), nnz, outTriplets,
	)
	_ = os.WriteFile(remapReport, []byte(rep), 0o644)

	fmt.Printf("[OK] REMAP: U=%d I=%d NNZ=%d\n", len(userIdx), len(itemIdx), nnz)
	fmt.Printf("  -> %s\n  -> %s\n  -> %s\n", outTriplets, userMapPath, itemMapPath)
}

func writeTripletsCSV(path string, buf []Triplet) error {
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

	_ = w.Write([]string{"uIdx", "iIdx", "rating"})
	for _, t := range buf {
		_ = w.Write([]string{
			strconv.Itoa(t.U),
			strconv.Itoa(t.I),
			strconv.FormatFloat(t.R, 'f', -1, 64),
		})
	}
	return nil
}

func writeUserMap(path string, m map[int]int) error {
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
	_ = w.Write([]string{"userId", "uIdx"})
	// Orden estable (por uIdx)
	type kv struct{ id, idx int }
	arr := make([]kv, 0, len(m))
	for id, idx := range m {
		arr = append(arr, kv{id, idx})
	}
	sort.Slice(arr, func(a, b int) bool { return arr[a].idx < arr[b].idx })
	for _, kv := range arr {
		_ = w.Write([]string{strconv.Itoa(kv.id), strconv.Itoa(kv.idx)})
	}
	return nil
}
func writeItemMap(path string, m map[int]int) error {
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
	_ = w.Write([]string{"movieId", "iIdx"})
	type kv struct{ id, idx int }
	arr := make([]kv, 0, len(m))
	for id, idx := range m {
		arr = append(arr, kv{id, idx})
	}
	sort.Slice(arr, func(a, b int) bool { return arr[a].idx < arr[b].idx })
	for _, kv := range arr {
		_ = w.Write([]string{strconv.Itoa(kv.id), strconv.Itoa(kv.idx)})
	}
	return nil
}
