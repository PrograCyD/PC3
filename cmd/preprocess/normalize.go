//go:build normalize
// +build normalize

package main

/*
NORMALIZACIÓN + CONSTRUCCIÓN CSR (centrado por usuario)

Entrada:
  - artifacts/ratings_ui.csv  // (uIdx,iIdx,rating)

Salidas:
  - artifacts/user_means.csv                   (uIdx,mean)
  - artifacts/matrix_user_csr/indptr.bin      (int64, len=U+1)
  - artifacts/matrix_user_csr/indices.bin     (int32, len=NNZ)
  - artifacts/matrix_user_csr/data.bin        (float32, len=NNZ)  // r' = r - mean(u)
  - artifacts/matrix_user_csr/meta.json       {U,I,NNZ,dtypes}
  - artifacts/normalize_report.txt            // resumen
*/

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"unsafe"
)

const (
	inTriplets = "artifacts/ratings_ui.csv"
	outMeans   = "artifacts/user_means.csv"
	outCSRDir  = "artifacts/matrix_user_csr"
	outIndptr  = "artifacts/matrix_user_csr/indptr.bin"
	outIndices = "artifacts/matrix_user_csr/indices.bin"
	outData    = "artifacts/matrix_user_csr/data.bin"
	outMeta    = "artifacts/matrix_user_csr/meta.json"
	normReport = "artifacts/normalize_report.txt"
)

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
	if err := os.MkdirAll(outCSRDir, 0o755); err != nil {
		fmt.Printf("ERROR creando %s: %v\n", outCSRDir, err)
		return
	}

	// PASO 1: leer triplets y calcular U, I, NNZ, y acumular sumas y counts por uIdx
	type row struct {
		u, i int
		r    float64
	}
	rows := make([]row, 0, 1_000_000)

	f, err := os.Open(inTriplets)
	if err != nil {
		fmt.Printf("ERROR abriendo %s: %v\n", inTriplets, err)
		return
	}
	rd := csv.NewReader(bufio.NewReader(f))
	rd.FieldsPerRecord = -1
	_, _ = rd.Read() // header

	var U, I, NNZ int
	// como viene ordenado por uIdx, iremos detectando máximos
	userSum := make(map[int]float64, 200000)
	userCnt := make(map[int]int, 200000)

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

		rows = append(rows, row{u, i, r})
		NNZ++
		if u+1 > U {
			U = u + 1
		}
		if i+1 > I {
			I = i + 1
		}
		userSum[u] += r
		userCnt[u]++
	}
	f.Close()

	// PASO 2: medias por usuario
	if err := writeMeans(outMeans, userSum, userCnt, U); err != nil {
		fmt.Printf("ERROR escribiendo medias: %v\n", err)
		return
	}

	means := make([]float64, U)
	for u := 0; u < U; u++ {
		if c := userCnt[u]; c > 0 {
			means[u] = userSum[u] / float64(c)
		}
	}

	// PASO 3: construir CSR (indptr, indices, data)
	indptr := make([]int64, U+1)
	indices := make([]int32, NNZ)
	data := make([]float32, NNZ)

	// Como rows está ordenado por u (viene de remap), podemos llenar en una pasada
	var currU int
	var pos int
	for _, rw := range rows {
		for currU <= rw.u {
			indptr[currU] = int64(pos)
			currU++
			if currU > rw.u {
				break
			}
		}
		indices[pos] = int32(rw.i)
		data[pos] = float32(rw.r - means[rw.u]) // centrado por usuario
		pos++
	}
	// cerrar último usuario
	for currU <= U {
		indptr[currU] = int64(pos)
		currU++
	}

	// PASO 4: persistir binarios + meta + reporte
	if err := writeBin(outIndptr, indptr); err != nil {
		fmt.Printf("ERROR indptr: %v\n", err)
		return
	}
	if err := writeBin(outIndices, indices); err != nil {
		fmt.Printf("ERROR indices: %v\n", err)
		return
	}
	if err := writeBin(outData, data); err != nil {
		fmt.Printf("ERROR data: %v\n", err)
		return
	}

	mt := meta{Users: U, Items: I, NNZ: NNZ}
	mt.DTypes.Indptr = "int64"
	mt.DTypes.Indices = "int32"
	mt.DTypes.Data = "float32"
	jb, _ := json.MarshalIndent(mt, "", "  ")
	_ = os.WriteFile(outMeta, jb, 0o644)

	rep := fmt.Sprintf(
		"== NORMALIZE + CSR ==\nU=%d I=%d NNZ=%d\nout=%s\n", U, I, NNZ, outCSRDir,
	)
	_ = os.WriteFile(normReport, []byte(rep), 0o644)

	fmt.Printf("[OK] CSR centrado por usuario creado: U=%d I=%d NNZ=%d\n", U, I, NNZ)
	fmt.Printf("  indptr=%s\n  indices=%s\n  data=%s\n  meta=%s\n", outIndptr, outIndices, outData, outMeta)
}

func writeMeans(path string, sum map[int]float64, cnt map[int]int, U int) error {
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
	_ = w.Write([]string{"uIdx", "mean"})
	for u := 0; u < U; u++ {
		var mean float64
		if c := cnt[u]; c > 0 {
			mean = sum[u] / float64(c)
		}
		_ = w.Write([]string{strconv.Itoa(u), strconv.FormatFloat(mean, 'f', -1, 64)})
	}
	return nil
}

// writeBin escribe cualquier slice primitivo en binario sin encabezado.
func writeBin[T ~int64 | ~int32 | ~float32](path string, arr []T) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	// conversión a bytes sin copiar: iteramos y escribimos cada elemento
	// (simple y seguro para tipos fijos)
	for _, v := range arr {
		// escribe en little-endian estándar de Go
		var buf [8]byte
		switch any(v).(type) {
		case int64:
			x := any(v).(int64)
			for i := 0; i < 8; i++ {
				buf[i] = byte(x >> (8 * i))
			}
			_, err = f.Write(buf[:8])
		case int32:
			x := any(v).(int32)
			for i := 0; i < 4; i++ {
				buf[i] = byte(x >> (8 * i))
			}
			_, err = f.Write(buf[:4])
		case float32:
			x := any(v).(float32)
			u := mathFloat32bits(x)
			for i := 0; i < 4; i++ {
				buf[i] = byte(u >> (8 * i))
			}
			_, err = f.Write(buf[:4])
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func mathFloat32bits(f float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&f))
}
