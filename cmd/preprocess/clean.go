//go:build clean
// +build clean

package main

/*
LIMPIEZA / INSPECCIÓN + FILTRADO (conservar películas con ≥5 ratings)

Objetivo:
- Explorar y diagnosticar el dataset (sin modificar datos) y,
- Aplicar limpieza efectiva reteniendo solo filas de películas con ≥5 ratings,
  generando artifacts/ratings_min5.csv y un reporte de filtrado.

Tareas:
1) Inspección (igual que antes):
   - Nulos / campos vacíos
   - Rango de rating [0.5, 5.0] con paso 0.5
   - Duplicados (userId, movieId) contiguos
   - Distribuciones e insights
   - Reporte: artifacts/clean_report.txt

2) Filtrado real (NUEVO):
   - Contar ratings por movieId (1ra pasada)
   - Escribir solo filas con movieId que cumplan ≥5 ratings (2da pasada)
   - Guardar CSV limpio: artifacts/ratings_min5.csv
   - Guardar reporte filtrado: artifacts/clean_filter_report.txt
   - Imprimir resumen (filas/películas eliminadas, usuarios retenidos, justificación)

*/

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"pc3/utils"
)

const (
	// Entrada
	ratingsPath = "data/ratings.csv"
	moviesPath  = "data/movies.csv"

	// Reporte de inspección
	reportPath = "artifacts/clean_report.txt"

	// Salidas del filtrado real
	filteredPath   = "artifacts/ratings_min5.csv"
	filterReport   = "artifacts/clean_filter_report.txt"
	minItemRatings = 5 // criterio fijo: conservar películas con ≥5 ratings
)

func main() {
	log := utils.NewLogger(true)
	timer := utils.NewTimer()

	// Asegurar directorio de artifacts para ambos reportes y el CSV limpio
	if err := os.MkdirAll("artifacts", 0o755); err != nil {
		log.Error("no se pudo crear artifacts/: %v", err)
		return
	}

	// ==================== ETAPA 1: INSPECCIÓN ====================
	log.Info("Inicio de inspección…")
	stats, err := inspectRatings(ratingsPath, log)
	if err != nil {
		log.Error("error inspeccionando ratings: %v", err)
		return
	}

	totalMovies, err := countMovies(moviesPath, log)
	if err != nil {
		log.Warn("no se pudo contar movies (%v) — no es crítico para esta etapa", err)
	}

	// Consola + reporte
	printConsoleSummary(stats, totalMovies, log)
	if err := writeReport(stats, totalMovies, reportPath); err != nil {
		log.Error("no se pudo escribir reporte de inspección: %v", err)
		return
	}

	// ==================== ETAPA 2: FILTRADO REAL (≥5) ====================
	if err := filterByPopularity(log); err != nil {
		log.Error("falló el filtrado real: %v", err)
		return
	}

	log.Info("Listo. Reportes en artifacts/. Tiempo total: %s", timer.Elapsed())
}

// ----- Estructuras de resumen (inspección) -----

type RatingsStats struct {
	TotalRows        int64
	TotalValidRows   int64
	TotalUsers       int
	TotalItems       int
	NullRows         int64
	OutOfRangeRows   int64
	NonStepRows      int64
	Duplicates       int64
	RatingBuckets    map[float64]int64 // 0.5, 1.0, …, 5.0
	StarBuckets      map[int]int64     // 1,2,3,4,5 (estrellas enteras)
	UserActivity     map[int]int       // userId -> count
	ItemActivity     map[int]int       // movieId -> count
	UserBuckets      map[string]int    // actividad por usuario (rangos)
	ItemBuckets      map[string]int    // actividad por ítem (rangos)
	UserMin, UserMax int
	ItemMin, ItemMax int
	UserAvg, ItemAvg float64

	// Insights / umbrales
	UsersLt5   int
	UsersGe100 int
	ItemsLt5   int
	ItemsLt10  int
	ItemsGe100 int
}

func newRatingsStats() *RatingsStats {
	buckets := map[float64]int64{}
	for r := 0.5; r <= 5.0+1e-9; r += 0.5 {
		buckets[roundHalf(r)] = 0
	}
	star := map[int]int64{1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
	return &RatingsStats{
		RatingBuckets: buckets,
		StarBuckets:   star,
		UserActivity:  make(map[int]int, 200000),
		ItemActivity:  make(map[int]int, 70000),
	}
}

func roundHalf(x float64) float64 {
	s := fmt.Sprintf("%.1f", x) // evita errores de flotantes
	v, _ := strconv.ParseFloat(s, 64)
	return v
}

func validStep(r float64) bool {
	const eps = 1e-9
	halfSteps := r / 0.5
	return mathAbs(halfSteps-mathRound(halfSteps)) < eps
}

func mathRound(x float64) float64 {
	if x >= 0 {
		return float64(int64(x + 0.5))
	}
	return float64(int64(x - 0.5))
}

func mathAbs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// ----- Lectura e inspección de ratings -----

func inspectRatings(path string, log *utils.Logger) (*RatingsStats, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("abrir %s: %w", path, err)
	}
	defer f.Close()

	reader := csv.NewReader(bufio.NewReader(f))
	reader.FieldsPerRecord = -1 // acepta filas con comas entrecomilladas
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("leer cabecera: %w", err)
	}
	if len(header) < 4 {
		return nil, errors.New("cabecera inesperada en ratings.csv (se esperan 4 columnas)")
	}

	stats := newRatingsStats()

	// Detector de duplicados con baja memoria (consecutivos)
	var prevUser, prevItem int
	var havePrev bool

	rowIdx := int64(0)
	for {
		row, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			// otras anormalidades: cuenta como nulo y sigue
			stats.NullRows++
			continue
		}
		rowIdx++

		if len(row) < 4 {
			stats.NullRows++
			continue
		}

		uidStr := strings.TrimSpace(row[0])
		iidStr := strings.TrimSpace(row[1])
		rStr := strings.TrimSpace(row[2])

		if uidStr == "" || iidStr == "" || rStr == "" {
			stats.NullRows++
			continue
		}

		uid, err1 := strconv.Atoi(uidStr)
		iid, err2 := strconv.Atoi(iidStr)
		r, err3 := strconv.ParseFloat(rStr, 64)

		if err1 != nil || err2 != nil || err3 != nil {
			stats.NullRows++
			continue
		}

		// rango y pasos
		if r < 0.5 || r > 5.0 {
			stats.OutOfRangeRows++
		} else if !validStep(r) {
			stats.NonStepRows++
		} else {
			stats.TotalValidRows++
			stats.RatingBuckets[roundHalf(r)]++
			// estrellas enteras (e.g., 3.5 se suma a 4)
			star := int(mathRound(r))
			if star < 1 {
				star = 1
			}
			if star > 5 {
				star = 5
			}
			stats.StarBuckets[star]++

			// actividad
			stats.UserActivity[uid]++
			stats.ItemActivity[iid]++
		}

		// duplicados (consecutivos)
		if havePrev && uid == prevUser && iid == prevItem {
			stats.Duplicates++
		}
		prevUser, prevItem, havePrev = uid, iid, true

		stats.TotalRows++
		if rowIdx%2_000_000 == 0 {
			log.Info("leídas ~%d filas…", rowIdx)
		}
	}

	// Estadísticos de actividad por usuario/ítem
	stats.TotalUsers = len(stats.UserActivity)
	stats.TotalItems = len(stats.ItemActivity)
	stats.UserMin, stats.UserMax, stats.UserAvg, stats.UserBuckets = summarizeActivity(stats.UserActivity)
	stats.ItemMin, stats.ItemMax, stats.ItemAvg, stats.ItemBuckets = summarizeActivity(stats.ItemActivity)

	// Insights / umbrales
	stats.UsersLt5, stats.UsersGe100 = countThresholds(stats.UserActivity, 5, 100)
	tmpLt5, tmpLt10, tmpGe100 := countThresholdsItems(stats.ItemActivity, 5, 10, 100)
	stats.ItemsLt5, stats.ItemsLt10, stats.ItemsGe100 = tmpLt5, tmpLt10, tmpGe100

	return stats, nil
}

func summarizeActivity(m map[int]int) (min int, max int, avg float64, buckets map[string]int) {
	if len(m) == 0 {
		return 0, 0, 0.0, map[string]int{}
	}
	min = 1<<31 - 1
	max = -1
	var sum int
	for _, c := range m {
		if c < min {
			min = c
		}
		if c > max {
			max = c
		}
		sum += c
	}
	avg = float64(sum) / float64(len(m))
	// buckets para informe
	buckets = map[string]int{
		"0-4":   0,
		"5-9":   0,
		"10-19": 0,
		"20-49": 0,
		"50-99": 0,
		"100+":  0,
	}
	for _, c := range m {
		switch {
		case c <= 4:
			buckets["0-4"]++
		case c <= 9:
			buckets["5-9"]++
		case c <= 19:
			buckets["10-19"]++
		case c <= 49:
			buckets["20-49"]++
		case c <= 99:
			buckets["50-99"]++
		default:
			buckets["100+"]++
		}
	}
	return
}

func countMovies(path string, log *utils.Logger) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, fmt.Errorf("abrir %s: %w", path, err)
	}
	defer f.Close()
	reader := csv.NewReader(bufio.NewReader(f))
	_, err = reader.Read() // header
	if err != nil {
		return 0, fmt.Errorf("leer cabecera: %w", err)
	}
	count := 0
	for {
		_, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return 0, err
		}
		count++
	}
	return count, nil
}

func countThresholds(m map[int]int, lt int, ge int) (ltCount int, geCount int) {
	for _, c := range m {
		if c < lt {
			ltCount++
		}
		if c >= ge {
			geCount++
		}
	}
	return
}

func countThresholdsItems(m map[int]int, ltA int, ltB int, geC int) (ltACount, ltBCount, geCCount int) {
	for _, c := range m {
		if c < ltA {
			ltACount++
		}
		if c < ltB {
			ltBCount++
		}
		if c >= geC {
			geCCount++
		}
	}
	return
}

// ----- Reporte / Consola (inspección) -----

func printConsoleSummary(s *RatingsStats, totalMovies int, log *utils.Logger) {
	log.Info("Resumen general:")
	log.Info("  Filas totales: %d | válidas: %d | nulas: %d | fuera de rango: %d | pasos inválidos: %d | duplicados: %d",
		s.TotalRows, s.TotalValidRows, s.NullRows, s.OutOfRangeRows, s.NonStepRows, s.Duplicates)
	log.Info("  Usuarios distintos (con al menos 1 rating): %d", s.TotalUsers)
	log.Info("  Películas distintas con rating: %d (en movies.csv hay %d)", s.TotalItems, totalMovies)

	log.Info("Distribución de ratings (paso 0.5):")
	keys := make([]float64, 0, len(s.RatingBuckets))
	for k := range s.RatingBuckets {
		keys = append(keys, k)
	}
	sort.Float64s(keys)
	for _, k := range keys {
		log.Info("  rating %.1f : %d", k, s.RatingBuckets[k])
	}

	log.Info("Distribución por estrellas enteras (1–5):")
	for star := 1; star <= 5; star++ {
		log.Info("  %d★ : %d", star, s.StarBuckets[star])
	}

	log.Info("Actividad por usuario (número de películas calificadas por usuario):")
	log.Info("  min=%d | max=%d | avg=%.2f", s.UserMin, s.UserMax, s.UserAvg)
	log.Info("  Buckets = usuarios que calificaron dentro del rango indicado:")
	printBucketsExplained("Usuarios por buckets", s.UserBuckets, log)

	log.Info("Actividad por película (número de ratings por película):")
	log.Info("  min=%d | max=%d | avg=%.2f", s.ItemMin, s.ItemMax, s.ItemAvg)
	log.Info("  Buckets = películas que recibieron una cantidad de ratings dentro del rango indicado:")
	printBucketsExplained("Ítems por buckets", s.ItemBuckets, log)

	// Insights / umbrales
	uPctLt5 := 100.0 * float64(s.UsersLt5) / float64(max(1, s.TotalUsers))
	uPctGe100 := 100.0 * float64(s.UsersGe100) / float64(max(1, s.TotalUsers))
	iPctLt5 := 100.0 * float64(s.ItemsLt5) / float64(max(1, s.TotalItems))
	iPctLt10 := 100.0 * float64(s.ItemsLt10) / float64(max(1, s.TotalItems))
	iPctGe100 := 100.0 * float64(s.ItemsGe100) / float64(max(1, s.TotalItems))

	log.Info("Insights / umbrales sugeridos:")
	log.Info("  Usuarios con <5 ratings: %d (%.2f%%) — en ML-25M debería ser ~0 porque ya filtran <20.", s.UsersLt5, uPctLt5)
	log.Info("  Usuarios con ≥100 ratings: %d (%.2f%%).", s.UsersGe100, uPctGe100)
	log.Info("  Películas con <5 ratings: %d (%.2f%%).", s.ItemsLt5, iPctLt5)
	log.Info("  Películas con <10 ratings: %d (%.2f%%).", s.ItemsLt10, iPctLt10)
	log.Info("  Películas con ≥100 ratings: %d (%.2f%%).", s.ItemsGe100, iPctGe100)
}

func writeReport(s *RatingsStats, totalMovies int, out string) error {
	var b strings.Builder
	fmt.Fprintf(&b, "== INSPECCIÓN MovieLens 25M ==\n\n")
	fmt.Fprintf(&b, "Filas totales          : %d\n", s.TotalRows)
	fmt.Fprintf(&b, "Filas válidas          : %d\n", s.TotalValidRows)
	fmt.Fprintf(&b, "Nulos                  : %d\n", s.NullRows)
	fmt.Fprintf(&b, "Fuera de rango         : %d\n", s.OutOfRangeRows)
	fmt.Fprintf(&b, "Pasos inválidos        : %d\n", s.NonStepRows)
	fmt.Fprintf(&b, "Duplicados (consecut.) : %d\n\n", s.Duplicates)

	fmt.Fprintf(&b, "Usuarios distintos     : %d\n", s.TotalUsers)
	fmt.Fprintf(&b, "Películas (ratings)    : %d\n", s.TotalItems)
	fmt.Fprintf(&b, "Películas (movies.csv) : %d\n\n", totalMovies)

	fmt.Fprintf(&b, "-- Distribución de ratings (paso 0.5) --\n")
	keys := make([]float64, 0, len(s.RatingBuckets))
	for k := range s.RatingBuckets {
		keys = append(keys, k)
	}
	sort.Float64s(keys)
	for _, k := range keys {
		fmt.Fprintf(&b, "  %.1f : %d\n", k, s.RatingBuckets[k])
	}
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "-- Distribución por estrellas enteras (1–5) --\n")
	for star := 1; star <= 5; star++ {
		fmt.Fprintf(&b, "  %d★ : %d\n", star, s.StarBuckets[star])
	}
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "-- Actividad por usuario (número de películas calificadas por usuario) --\n")
	fmt.Fprintf(&b, "min=%d max=%d avg=%.2f\n", s.UserMin, s.UserMax, s.UserAvg)
	writeBuckets(&b, s.UserBuckets)
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "-- Actividad por película (número de ratings por película) --\n")
	fmt.Fprintf(&b, "min=%d max=%d avg=%.2f\n", s.ItemMin, s.ItemMax, s.ItemAvg)
	writeBuckets(&b, s.ItemBuckets)
	fmt.Fprintf(&b, "\n")

	// Insights / umbrales
	uPctLt5 := 100.0 * float64(s.UsersLt5) / float64(max(1, s.TotalUsers))
	uPctGe100 := 100.0 * float64(s.UsersGe100) / float64(max(1, s.TotalUsers))
	iPctLt5 := 100.0 * float64(s.ItemsLt5) / float64(max(1, s.TotalItems))
	iPctLt10 := 100.0 * float64(s.ItemsLt10) / float64(max(1, s.TotalItems))
	iPctGe100 := 100.0 * float64(s.ItemsGe100) / float64(max(1, s.TotalItems))

	fmt.Fprintf(&b, "-- Insights / umbrales sugeridos --\n")
	fmt.Fprintf(&b, "Usuarios con <5 ratings   : %d (%.2f%%) — en ML-25M debería ser ~0.\n", s.UsersLt5, uPctLt5)
	fmt.Fprintf(&b, "Usuarios con ≥100 ratings : %d (%.2f%%)\n", s.UsersGe100, uPctGe100)
	fmt.Fprintf(&b, "Películas con <5 ratings  : %d (%.2f%%)\n", s.ItemsLt5, iPctLt5)
	fmt.Fprintf(&b, "Películas con <10 ratings : %d (%.2f%%)\n", s.ItemsLt10, iPctLt10)
	fmt.Fprintf(&b, "Películas con ≥100 ratings: %d (%.2f%%)\n", s.ItemsGe100, iPctGe100)

	if err := os.WriteFile(out, []byte(b.String()), 0o644); err != nil {
		return err
	}
	return nil
}

func printBucketsExplained(label string, m map[string]int, log *utils.Logger) {
	ordered := []string{"0-4", "5-9", "10-19", "20-49", "50-99", "100+"}
	log.Info("%s (interpretación):", label)
	log.Info("  0-4    = cantidad de entidades con entre 0 y 4 registros (usuarios: películas calificadas; ítems: ratings recibidos)")
	log.Info("  5-9    = … entre 5 y 9")
	log.Info("  10-19  = … entre 10 y 19")
	log.Info("  20-49  = … entre 20 y 49")
	log.Info("  50-99  = … entre 50 y 99")
	log.Info("  100+   = … 100 o más")
	for _, k := range ordered {
		log.Info("    %-5s : %d", k, m[k])
	}
}

func writeBuckets(b *strings.Builder, m map[string]int) {
	labels := []string{"0-4", "5-9", "10-19", "20-49", "50-99", "100+"}
	for _, k := range labels {
		fmt.Fprintf(b, "  %-6s : %d\n", k, m[k])
	}
}

// utilidades pequeñas
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ====================== ETAPA 2: FILTRADO REAL (≥5) ======================

func filterByPopularity(log *utils.Logger) error {
	log.Info("=== FILTRADO REAL: conservar películas con ≥%d ratings ===", minItemRatings)

	// 1) Conteo por movieId (1ª pasada)
	counts, totalRows, err := countPerItem(ratingsPath)
	if err != nil {
		return fmt.Errorf("conteo por ítem falló: %v", err)
	}
	distinctItems := len(counts)

	keptItems, droppedItems := 0, 0
	for _, c := range counts {
		if c >= minItemRatings {
			keptItems++
		} else {
			droppedItems++
		}
	}

	// 2) Escritura filtrada (2ª pasada)
	if err := os.MkdirAll(filepath.Dir(filteredPath), 0o755); err != nil {
		return fmt.Errorf("crear dir salida: %w", err)
	}
	keptRows, keptUsers, err := writeFilteredRatings(ratingsPath, filteredPath, counts, minItemRatings)
	if err != nil {
		return fmt.Errorf("escritura del CSV filtrado falló: %v", err)
	}
	droppedRows := totalRows - keptRows

	// 3) Reporte de filtrado
	if err := writeFilterReport(filterReport, totalRows, keptRows, droppedRows,
		distinctItems, keptItems, droppedItems, keptUsers); err != nil {
		return fmt.Errorf("no se pudo escribir el reporte de filtrado: %v", err)
	}

	// 4) Consola (resumen)
	log.Info("=== RESUMEN FILTRADO ===")
	log.Info("Criterio: conservar películas con ≥%d ratings (estabilidad de similitud y reducción de ruido).", minItemRatings)
	log.Info("Filas originales     : %d", totalRows)
	log.Info("Filas retenidas      : %d", keptRows)
	log.Info("Filas eliminadas     : %d (%.2f%%)", droppedRows, percent64(droppedRows, totalRows))
	log.Info("Películas totales    : %d", distinctItems)
	log.Info("Películas retenidas  : %d", keptItems)
	log.Info("Películas eliminadas : %d (%.2f%%)", droppedItems, percent(droppedItems, distinctItems))
	log.Info("Usuarios en limpio   : %d (usuarios con ≥1 rating retenido)", keptUsers)
	log.Info("Archivo limpio       : %s", filteredPath)
	log.Info("Reporte de filtrado  : %s", filterReport)

	return nil
}

func countPerItem(path string) (map[int]int, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("abrir %s: %w", path, err)
	}
	defer f.Close()

	reader := csv.NewReader(bufio.NewReader(f))
	reader.FieldsPerRecord = -1
	_, err = reader.Read() // cabecera
	if err != nil {
		return nil, 0, fmt.Errorf("leer cabecera: %w", err)
	}

	itemCount := make(map[int]int, 70000)
	var total int64

	for {
		row, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			continue
		}
		if len(row) < 4 {
			continue
		}
		iidStr := strings.TrimSpace(row[1])
		iid, err := strconv.Atoi(iidStr)
		if err != nil {
			continue
		}
		itemCount[iid]++
		total++
	}
	return itemCount, total, nil
}

func writeFilteredRatings(inPath, outPath string, counts map[int]int, minRatings int) (int64, int, error) {
	inF, err := os.Open(inPath)
	if err != nil {
		return 0, 0, fmt.Errorf("abrir %s: %w", inPath, err)
	}
	defer inF.Close()

	outF, err := os.Create(outPath)
	if err != nil {
		return 0, 0, fmt.Errorf("crear %s: %w", outPath, err)
	}
	defer outF.Close()

	reader := csv.NewReader(bufio.NewReader(inF))
	reader.FieldsPerRecord = -1
	writer := csv.NewWriter(bufio.NewWriter(outF))
	defer writer.Flush()

	header, err := reader.Read()
	if err != nil {
		return 0, 0, fmt.Errorf("leer cabecera: %w", err)
	}
	if len(header) < 4 {
		return 0, 0, errors.New("cabecera inesperada en ratings.csv (se esperan 4 columnas)")
	}
	if err := writer.Write(header); err != nil {
		return 0, 0, fmt.Errorf("escribir cabecera: %w", err)
	}

	userSeen := make(map[int]struct{}, 200000)
	var keptRows int64

	for {
		row, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			continue
		}
		if len(row) < 4 {
			continue
		}
		uidStr := strings.TrimSpace(row[0])
		iidStr := strings.TrimSpace(row[1])

		iid, err := strconv.Atoi(iidStr)
		if err != nil {
			continue
		}
		if counts[iid] >= minRatings {
			if err := writer.Write(row); err != nil {
				return keptRows, len(userSeen), fmt.Errorf("escribir fila: %w", err)
			}
			keptRows++
			if uid, err := strconv.Atoi(uidStr); err == nil {
				userSeen[uid] = struct{}{}
			}
		}
	}

	return keptRows, len(userSeen), nil
}

func writeFilterReport(path string, totalRows, keptRows, droppedRows int64,
	distinctItems, keptItems, droppedItems, keptUsers int) error {

	var b strings.Builder
	fmt.Fprintf(&b, "== FILTRADO MovieLens 25M ==\n\n")
	fmt.Fprintf(&b, "Criterio aplicado: conservar películas con ≥%d ratings.\n\n", minItemRatings)
	fmt.Fprintf(&b, "Filas originales     : %d\n", totalRows)
	fmt.Fprintf(&b, "Filas retenidas      : %d\n", keptRows)
	fmt.Fprintf(&b, "Filas eliminadas     : %d (%.2f%%)\n\n", droppedRows, percent64(droppedRows, totalRows))

	fmt.Fprintf(&b, "Películas totales    : %d\n", distinctItems)
	fmt.Fprintf(&b, "Películas retenidas  : %d\n", keptItems)
	fmt.Fprintf(&b, "Películas eliminadas : %d (%.2f%%)\n\n", droppedItems, percent(droppedItems, distinctItems))

	fmt.Fprintf(&b, "Usuarios en limpio   : %d (usuarios con ≥1 rating retenido)\n\n", keptUsers)

	fmt.Fprintf(&b, "Justificación del umbral:\n")
	fmt.Fprintf(&b, "- Con menos de %d ratings por película, coseno y Pearson son inestables (poco soporte conjunto).\n", minItemRatings)
	fmt.Fprintf(&b, "- Mantener solo ítems con suficiente señal reduce ruido y costo computacional.\n")
	fmt.Fprintf(&b, "- Este recorte es para el cómputo de similitudes; la UI puede seguir mostrando metadata completa de movies.\n")

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func percent(part, total int) float64 {
	if total <= 0 {
		return 0
	}
	return 100.0 * float64(part) / float64(total)
}
func percent64(part, total int64) float64 {
	if total <= 0 {
		return 0
	}
	return 100.0 * float64(part) / float64(total)
}
