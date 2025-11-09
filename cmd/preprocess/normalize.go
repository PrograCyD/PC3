//go:build normalize
// +build normalize

package main

/*
NORMALIZACIÓN + CSR por USUARIO e ÍTEM (opcionalmente ambas)

Entrada:
  - artifacts/ratings_ui.csv  // (uIdx,iIdx,rating) ordenado por uIdx

Salidas (según --axis):
  - artifacts/user_means.csv
  - artifacts/matrix_user_csr/{indptr.bin,indices.bin,data.bin,meta.json}

  - artifacts/item_means.csv
  - artifacts/matrix_item_csr/{indptr.bin,indices.bin,data.bin,meta.json}

Notas:
  - Pearson(user-based) usa matrix_user_csr (centrado por usuario).
  - Pearson(item-based) usa matrix_item_csr (centrado por ítem).
  - Coseno item-based puede seguir usando ratings_ui.csv (no requiere centrar).
*/

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"unsafe"
)

const (
	inTriplets = "artifacts/ratings_ui.csv"

	// USER
	userMeansPath = "artifacts/user_means.csv"
	userDir       = "artifacts/matrix_user_csr"
	userIndptr    = "artifacts/matrix_user_csr/indptr.bin"
	userIndices   = "artifacts/matrix_user_csr/indices.bin"
	userData      = "artifacts/matrix_user_csr/data.bin"
	userMeta      = "artifacts/matrix_user_csr/meta.json"

	// ITEM
	itemMeansPath = "artifacts/item_means.csv"
	itemDir       = "artifacts/matrix_item_csr"
	itemIndptr    = "artifacts/matrix_item_csr/indptr.bin"
	itemIndices   = "artifacts/matrix_item_csr/indices.bin"
	itemData      = "artifacts/matrix_item_csr/data.bin"
	itemMeta      = "artifacts/matrix_item_csr/meta.json"
)

type trip struct {
	u, i int
	r    float64
}

type meta struct {
	Users  int `json:"users"`
	Items  int `json:"items"`
	NNZ    int `json:"nnz"`
	DTypes struct {
		Indptr  string `json:"indptr"`
		Indices string `json:"indices"`
		Data    string `json:"data"`
	} `json:"dtypes"`
}

func main() {
	var axis string
	flag.StringVar(&axis, "axis", "both", "user | item | both")
	flag.Parse()

	// --- PASO 1: cargar triplets una vez y colectar tamaños ---
	f, err := os.Open(inTriplets)
	if err != nil {
		fmt.Printf("ERROR abriendo %s: %v\n", inTriplets, err)
		return
	}
	defer f.Close()
	rd := csv.NewReader(bufio.NewReader(f))
	_, _ = rd.Read() // header

	rows := make([]trip, 0, 1_000_000)
	U, I, NNZ := 0, 0, 0
	for {
		rec, err := rd.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			continue
		}
		if len(rec) < 3 {
			continue
		}
		u, _ := strconv.Atoi(rec[0])
		i, _ := strconv.Atoi(rec[1])
		r, _ := strconv.ParseFloat(rec[2], 64)
		rows = append(rows, trip{u, i, r})
		NNZ++
		if u+1 > U {
			U = u + 1
		}
		if i+1 > I {
			I = i + 1
		}
	}

	// --- USER: medias + CSR centrado por usuario (sólo si aplica) ---
	if axis == "user" || axis == "both" {
		if err := os.MkdirAll(userDir, 0o755); err != nil {
			fmt.Printf("ERROR creando %s: %v\n", userDir, err)
			return
		}

		userSum := make([]float64, U)
		userCnt := make([]int, U)
		for _, t := range rows {
			userSum[t.u] += t.r
			userCnt[t.u]++
		}
		if err := writeMeansDense(userMeansPath, userSum, userCnt); err != nil {
			fmt.Printf("ERROR escribiendo user_means: %v\n", err)
			return
		}
		userMean := make([]float64, U)
		for u := 0; u < U; u++ {
			if userCnt[u] > 0 {
				userMean[u] = userSum[u] / float64(userCnt[u])
			}
		}

		// Como ratings_ui.csv viene ordenado por u, el CSR se arma en una pasada.
		indptr := make([]int64, U+1)
		indices := make([]int32, NNZ)
		data := make([]float32, NNZ)
		var currU, pos int
		for _, t := range rows {
			for currU <= t.u {
				indptr[currU] = int64(pos)
				currU++
				if currU > t.u {
					break
				}
			}
			indices[pos] = int32(t.i)
			data[pos] = float32(t.r - userMean[t.u])
			pos++
		}
		for currU <= U {
			indptr[currU] = int64(pos)
			currU++
		}

		if err := writeBin(userIndptr, indptr); err != nil {
			fmt.Println("ERROR user indptr:", err)
			return
		}
		if err := writeBin(userIndices, indices); err != nil {
			fmt.Println("ERROR user indices:", err)
			return
		}
		if err := writeBin(userData, data); err != nil {
			fmt.Println("ERROR user data:", err)
			return
		}

		mt := meta{Users: U, Items: I, NNZ: NNZ}
		mt.DTypes.Indptr, mt.DTypes.Indices, mt.DTypes.Data = "int64", "int32", "float32"
		jb, _ := json.MarshalIndent(mt, "", "  ")
		_ = os.WriteFile(userMeta, jb, 0o644)

		fmt.Printf("[OK] USER CSR -> U=%d I=%d NNZ=%d  out=%s\n", U, I, NNZ, userDir)
	}

	// --- ITEM: medias + CSR centrado por ítem (sólo si aplica) ---
	if axis == "item" || axis == "both" {
		if err := os.MkdirAll(itemDir, 0o755); err != nil {
			fmt.Printf("ERROR creando %s: %v\n", itemDir, err)
			return
		}

		itemSum := make([]float64, I)
		itemCnt := make([]int, I)
		for _, t := range rows {
			itemSum[t.i] += t.r
			itemCnt[t.i]++
		}
		if err := writeMeansDense(itemMeansPath, itemSum, itemCnt); err != nil {
			fmt.Printf("ERROR escribiendo item_means: %v\n", err)
			return
		}
		itemMean := make([]float64, I)
		for i := 0; i < I; i++ {
			if itemCnt[i] > 0 {
				itemMean[i] = itemSum[i] / float64(itemCnt[i])
			}
		}

		// Construir CSR por ítem (filas=ítems). Hacemos "contar y volcar":
		indptr := make([]int64, I+1)
		for i := 0; i < I; i++ {
			indptr[i+1] = indptr[i] + int64(itemCnt[i])
		}
		indices := make([]int32, NNZ) // aquí guardamos uIdx
		data := make([]float32, NNZ)  // r - mean(item)
		// cursores de escritura por ítem
		writePos := make([]int64, I)
		copy(writePos, indptr)

		for _, t := range rows {
			p := writePos[t.i]
			indices[p] = int32(t.u)
			data[p] = float32(t.r - itemMean[t.i])
			writePos[t.i]++
		}

		if err := writeBin(itemIndptr, indptr); err != nil {
			fmt.Println("ERROR item indptr:", err)
			return
		}
		if err := writeBin(itemIndices, indices); err != nil {
			fmt.Println("ERROR item indices:", err)
			return
		}
		if err := writeBin(itemData, data); err != nil {
			fmt.Println("ERROR item data:", err)
			return
		}

		mt := meta{Users: U, Items: I, NNZ: NNZ}
		mt.DTypes.Indptr, mt.DTypes.Indices, mt.DTypes.Data = "int64", "int32", "float32"
		jb, _ := json.MarshalIndent(mt, "", "  ")
		_ = os.WriteFile(itemMeta, jb, 0o644)

		fmt.Printf("[OK] ITEM CSR -> U=%d I=%d NNZ=%d  out=%s\n", U, I, NNZ, itemDir)
	}
}

// --- utilidades ---

func writeMeansDense(path string, sum []float64, cnt []int) error {
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
	_ = w.Write([]string{"idx", "mean"})
	for i := 0; i < len(sum); i++ {
		var m float64
		if cnt[i] > 0 {
			m = sum[i] / float64(cnt[i])
		}
		_ = w.Write([]string{strconv.Itoa(i), strconv.FormatFloat(m, 'f', -1, 64)})
	}
	return nil
}

// writeBin: guarda slices primitivos (int64, int32, float32) en little-endian
func writeBin[T ~int64 | ~int32 | ~float32](path string, arr []T) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var buf [8]byte
	for _, v := range arr {
		switch any(v).(type) {
		case int64:
			x := any(v).(int64)
			for i := 0; i < 8; i++ {
				buf[i] = byte(x >> (8 * i))
			}
			if _, err = f.Write(buf[:8]); err != nil {
				return err
			}
		case int32:
			x := any(v).(int32)
			for i := 0; i < 4; i++ {
				buf[i] = byte(x >> (8 * i))
			}
			if _, err = f.Write(buf[:4]); err != nil {
				return err
			}
		case float32:
			u := mathFloat32bits(any(v).(float32))
			for i := 0; i < 4; i++ {
				buf[i] = byte(u >> (8 * i))
			}
			if _, err = f.Write(buf[:4]); err != nil {
				return err
			}
		}
	}
	return nil
}

func mathFloat32bits(f float32) uint32 { return *(*uint32)(unsafe.Pointer(&f)) }
