//go:build recommend
// +build recommend

package main

/*
RECOMMEND + EVALUATION (secuencial, con cronometraje)
- Split hold-out por usuario (test_ratio).
- Predice con:
    * user-based + Pearson (usa user_topk_pearson.csv y user_means.csv)
    * item-based + Coseno  (usa item_topk_cosine.csv)
- Calcula MAE y RMSE.
- Mide tiempos por fase y escribe un reporte en artifacts/reports/.

Entradas:
  - artifacts/ratings_ui.csv
  - artifacts/sim/user_topk_pearson.csv   o   artifacts/sim/item_topk_cosine.csv
  - artifacts/user_means.csv  (solo para model=user)

Flags:
  --model=user|item
  --sim=path/to/sim.csv
  --test_ratio=0.1
  --k_eval=0        (si >0, límite de vecinos a usar)
  --report=""       (ruta opcional; por defecto artifacts/reports/recommend_<model>.txt)

Salida:
  - Consola: MAE, RMSE, tiempos y throughput
  - Reporte: artifacts/reports/recommend_<model>.txt
*/

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const tripletsPath = "artifacts/ratings_ui.csv"
const userMeansPath = "artifacts/user_means.csv"

type edge struct {
	to int
	w  float64
}
type ur struct {
	i int
	r float64
} // ratings por usuario
type ir struct {
	u int
	r float64
} // ratings por item

func main() {
	var model, simPath, reportPath string
	var testRatio float64
	var kEval int

	flag.StringVar(&model, "model", "user", "user | item")
	flag.StringVar(&simPath, "sim", "", "ruta del CSV de similitud")
	flag.Float64Var(&testRatio, "test_ratio", 0.1, "proporción de test por usuario")
	flag.IntVar(&kEval, "k_eval", 0, "si >0, límite de vecinos al predecir")
	flag.StringVar(&reportPath, "report", "", "ruta de reporte (opcional)")
	flag.Parse()
	if simPath == "" {
		panic("--sim requerido (ruta a user_topk_pearson.csv o item_topk_cosine.csv)")
	}
	if reportPath == "" {
		_ = os.MkdirAll("artifacts/reports", 0o755)
		reportPath = filepath.Join("artifacts", "reports", fmt.Sprintf("recommend_%s.txt", model))
	}

	t0 := time.Now()

	// 1) Cargar ratings en memoria
	users := make(map[int][]ur) // u -> [(i,r)]
	items := make(map[int][]ir) // i -> [(u,r)]
	f, _ := os.Open(tripletsPath)
	rd := csv.NewReader(bufio.NewReader(f))
	_, _ = rd.Read() // header
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
		users[u] = append(users[u], ur{i, r})
		items[i] = append(items[i], ir{u, r})
	}
	f.Close()
	tLoadRatings := time.Since(t0)

	// 2) Cargar similitudes
	sim := make(map[int][]edge) // nodo -> vecinos (ya ordenados)
	sf, _ := os.Open(simPath)
	sr := csv.NewReader(bufio.NewReader(sf))
	_, _ = sr.Read() // header
	for {
		rec, err := sr.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			continue
		}
		a, _ := strconv.Atoi(rec[0])
		b, _ := strconv.Atoi(rec[1])
		w, _ := strconv.ParseFloat(rec[2], 64)
		sim[a] = append(sim[a], edge{to: b, w: w})
	}
	sf.Close()
	tLoadSim := time.Since(t0) - tLoadRatings

	// 3) Medias de usuario (solo model=user)
	means := make(map[int]float64)
	var tLoadMeans time.Duration
	if model == "user" {
		m0 := time.Now()
		mf, _ := os.Open(userMeansPath)
		mr := csv.NewReader(bufio.NewReader(mf))
		_, _ = mr.Read()
		for {
			rec, err := mr.Read()
			if err != nil {
				if err.Error() == "EOF" {
					break
				}
				continue
			}
			u, _ := strconv.Atoi(rec[0])
			m, _ := strconv.ParseFloat(rec[1], 64)
			means[u] = m
		}
		mf.Close()
		tLoadMeans = time.Since(m0)
	}

	// 4) Split hold-out por usuario
	s0 := time.Now()
	rand.Seed(time.Now().UnixNano())
	type testPair struct {
		u, i int
		r    float64
	}
	var test []testPair
	train := make(map[int]map[int]float64) // u -> (i->r)

	for u, lst := range users {
		if len(lst) < 2 {
			continue
		} // necesita al menos 2 para train/test
		perm := rand.Perm(len(lst))
		szTest := int(math.Max(1, math.Round(testRatio*float64(len(lst)))))
		if szTest >= len(lst) {
			szTest = len(lst) - 1
		}
		tr := make(map[int]float64, len(lst)-szTest)
		for k, idx := range perm {
			it := lst[idx]
			if k < szTest {
				test = append(test, testPair{u: u, i: it.i, r: it.r})
			} else {
				tr[it.i] = it.r
			}
		}
		train[u] = tr
	}
	tSplit := time.Since(s0)

	// 5) Predicción y métricas (esto es el "tiempo de evaluación de métricas")
	p0 := time.Now()
	var absSum, sqSum float64
	var n int

	for _, t := range test {
		var pred float64
		if model == "user" {
			nu := sim[t.u]
			if kEval > 0 && len(nu) > kEval {
				nu = nu[:kEval]
			}
			var num, den float64
			for _, e := range nu {
				rv := ratingFromList(items[t.i], e.to) // rating de vecino e.to sobre item i
				if rv <= 0 {
					continue
				} // MovieLens ratings >= 0.5
				num += e.w * (rv - means[e.to])
				den += math.Abs(e.w)
			}
			if den == 0 {
				pred = means[t.u]
			} else {
				pred = means[t.u] + num/den
			}
			pred = clamp(pred, 0.5, 5.0)
		} else {
			ni := sim[t.i]
			if kEval > 0 && len(ni) > kEval {
				ni = ni[:kEval]
			}
			var num, den float64
			uj := train[t.u] // ratings de u en train
			for _, e := range ni {
				if rj, ok := uj[e.to]; ok {
					num += e.w * rj
					den += math.Abs(e.w)
				}
			}
			if den == 0 {
				pred = meanMap(uj)
			} else {
				pred = num / den
			}
			pred = clamp(pred, 0.5, 5.0)
		}
		err := t.r - pred
		absSum += math.Abs(err)
		sqSum += err * err
		n++
	}
	tPredict := time.Since(p0)
	mae := absSum / float64(n)
	rmse := math.Sqrt(sqSum / float64(n))
	throughput := float64(n) / tPredict.Seconds() // preds/s

	tTotal := time.Since(t0)

	// Consola
	fmt.Printf("[MODEL=%s] eval=%d  MAE=%.4f  RMSE=%.4f\n", strings.ToUpper(model), n, mae, rmse)
	fmt.Printf("Times: load_ratings=%s  load_sim=%s  load_means=%s  split=%s  predict=%s  TOTAL=%s\n",
		tLoadRatings, tLoadSim, tLoadMeans, tSplit, tPredict, tTotal)
	fmt.Printf("Throughput: %.0f preds/s (k_eval=%d)\n", throughput, kEval)

	// Reporte
	rep := fmt.Sprintf(
		`== RECOMMEND + EVAL (%s) ==
Sim CSV          : %s
Ratings CSV      : %s
User means       : %v
test_ratio       : %.2f
k_eval           : %d

Evaluated pairs  : %d
MAE              : %.4f
RMSE             : %.4f
Throughput       : %.0f preds/s

Tiempos:
  Cargar ratings : %s
  Cargar sim     : %s
  Cargar medias  : %s
  Split hold-out : %s
  Predecir       : %s
  TOTAL          : %s
`,
		strings.ToUpper(model), simPath, tripletsPath, model == "user",
		testRatio, kEval, n, mae, rmse, throughput,
		tLoadRatings, tLoadSim, tLoadMeans, tSplit, tPredict, tTotal,
	)
	_ = os.WriteFile(reportPath, []byte(rep), 0o644)
	fmt.Printf("Reporte -> %s\n", reportPath)
}

func ratingFromList(lst []ir, u int) float64 {
	for _, x := range lst {
		if x.u == u {
			return x.r
		}
	}
	return 0
}
func clamp(x, a, b float64) float64 {
	if x < a {
		return a
	}
	if x > b {
		return b
	}
	return x
}
func meanMap(m map[int]float64) float64 {
	if len(m) == 0 {
		return 3.0
	}
	var s float64
	for _, v := range m {
		s += v
	}
	return s / float64(len(m))
}
