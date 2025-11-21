//go:build recommend
// +build recommend

package main

/*
RECOMMEND + EVALUATION (secuencial, con cronometraje y métricas top-K)

- Split hold-out por usuario (test_ratio).
- Predice con:
    * user-based  (usa user_topk_*.csv y user_means.csv; ratings centrados)
    * item-based  (usa item_topk_*.csv; centrado opcional con --centered)
- Calcula:
    * MAE y RMSE (error de predicción)
    * Precision@K, Recall@K, NDCG@K, HitRate@K (métricas top-K por usuario)
- Mide tiempos por fase y escribe un reporte en artifacts/reports/.

Entradas:
  - artifacts/ratings_ui.csv
  - artifacts/sim/user_topk_*.csv   o   artifacts/sim/item_topk_*.csv
  - artifacts/user_means.csv  (solo para model=user)

Flags:
  --model=user|item
  --sim=path/to/sim.csv
  --test_ratio=0.1
  --k_eval=0        (si >0, límite de vecinos de similitud a usar en la predicción)
  --k_metrics=20    (K para métricas top-K: Precision@K, Recall@K, NDCG@K, HitRate@K)
  --rel_th=4.0      (rating mínimo para considerar un ítem relevante)
  --centered=false  (solo model=item; true si las similitudes se calcularon sobre ratings centrados)
  --report=""       (ruta opcional; por defecto artifacts/reports/recommend_<model>.txt)
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
	"sort"
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

// para métricas top-K
type evalRec struct {
	i     int
	rTrue float64
	rPred float64
}

func main() {
	var model, simPath, reportPath string
	var testRatio float64
	var kEval int
	var kMetrics int
	var relTh float64
	var centered bool // solo para item-based

	flag.StringVar(&model, "model", "user", "user | item")
	flag.StringVar(&simPath, "sim", "", "ruta del CSV de similitud")
	flag.Float64Var(&testRatio, "test_ratio", 0.1, "proporción de test por usuario")
	flag.IntVar(&kEval, "k_eval", 0, "si >0, límite de vecinos al predecir")
	flag.IntVar(&kMetrics, "k_metrics", 20, "K para métricas top-K (precision/recall/NDCG)")
	flag.Float64Var(&relTh, "rel_th", 4.0, "rating mínimo para considerar un ítem relevante")
	flag.BoolVar(&centered, "centered", false, "solo model=item: true si similitudes se calcularon sobre ratings centrados")
	flag.StringVar(&reportPath, "report", "", "ruta de reporte (opcional)")
	flag.Parse()

	if simPath == "" {
		panic("--sim requerido (ruta a user_topk_*.csv o item_topk_*.csv)")
	}
	if reportPath == "" {
		_ = os.MkdirAll("artifacts/reports", 0o755)
		reportPath = filepath.Join("artifacts", "reports", fmt.Sprintf("recommend_%s.txt", model))
	}

	t0 := time.Now()

	// -------------------------------------------------------------------------
	// 1) Cargar ratings en memoria
	// -------------------------------------------------------------------------
	users := make(map[int][]ur) // u -> [(i,r)]
	items := make(map[int][]ir) // i -> [(u,r)]

	f, err := os.Open(tripletsPath)
	if err != nil {
		panic(err)
	}
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

	// -------------------------------------------------------------------------
	// 2) Cargar similitudes
	// -------------------------------------------------------------------------
	sim := make(map[int][]edge) // nodo -> vecinos (ya ordenados)
	sf, err := os.Open(simPath)
	if err != nil {
		panic(err)
	}
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

	// -------------------------------------------------------------------------
	// 3) Medias de usuario (solo model=user)
	// -------------------------------------------------------------------------
	means := make(map[int]float64)
	var tLoadMeans time.Duration
	if model == "user" {
		m0 := time.Now()
		mf, err := os.Open(userMeansPath)
		if err != nil {
			panic(err)
		}
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

	// -------------------------------------------------------------------------
	// 4) Split hold-out por usuario
	// -------------------------------------------------------------------------
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

	// -------------------------------------------------------------------------
	// 5) Predicción y métricas de error (MAE, RMSE)
	//    + recopilación de datos para métricas top-K
	// -------------------------------------------------------------------------
	p0 := time.Now()
	var absSum, sqSum float64
	var n int

	evalByUser := make(map[int][]evalRec) // u -> lista de (i, rTrue, rPred)

	for _, t := range test {
		var pred float64

		if model == "user" {
			// USER-BASED: se asume que sim se calculó sobre ratings centrados (Pearson o Cosine centrado)
			nu := sim[t.u]
			if kEval > 0 && len(nu) > kEval {
				nu = nu[:kEval]
			}
			var num, den float64
			for _, e := range nu {
				rv := ratingFromList(items[t.i], e.to) // rating de vecino e.to sobre item i
				if rv <= 0 {
					continue
				}
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
			// ITEM-BASED: podemos usar fórmula centrada o no centrada
			ni := sim[t.i]
			if kEval > 0 && len(ni) > kEval {
				ni = ni[:kEval]
			}
			uj := train[t.u] // ratings de u en train

			if centered {
				// similitudes calculadas sobre ratings centrados:
				//   r'_u,j = r_u,j - mean_u
				// pred = mean_u + sum_j sim(i,j)*(r_u,j - mean_u) / sum_j |sim(i,j)|
				meanU := meanMap(uj)
				var num, den float64
				for _, e := range ni {
					if rj, ok := uj[e.to]; ok {
						num += e.w * (rj - meanU)
						den += math.Abs(e.w)
					}
				}
				if den == 0 {
					pred = meanU
				} else {
					pred = meanU + num/den
				}
			} else {
				// similitudes calculadas sobre ratings sin centrar (por ejemplo Cosine/Jaccard item)
				// pred = sum_j sim(i,j)*r_u,j / sum_j |sim(i,j)|
				var num, den float64
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
			}

			pred = clamp(pred, 0.5, 5.0)
		}

		err := t.r - pred
		absSum += math.Abs(err)
		sqSum += err * err
		n++

		// guardar para métricas top-K
		evalByUser[t.u] = append(evalByUser[t.u], evalRec{
			i:     t.i,
			rTrue: t.r,
			rPred: pred,
		})
	}
	tPredict := time.Since(p0)

	mae := absSum / float64(n)
	rmse := math.Sqrt(sqSum / float64(n))
	throughput := float64(n) / tPredict.Seconds() // preds/s

	tTotal := time.Since(t0)

	// -------------------------------------------------------------------------
	// 6) Métricas top-K (Precision@K, Recall@K, NDCG@K, HitRate@K)
	// -------------------------------------------------------------------------
	precK, recK, ndcgK, hitRateK := computeTopKMetrics(evalByUser, kMetrics, relTh)

	// -------------------------------------------------------------------------
	// 7) Consola
	// -------------------------------------------------------------------------
	fmt.Printf("[MODEL=%s] eval=%d  MAE=%.4f  RMSE=%.4f\n",
		strings.ToUpper(model), n, mae, rmse)
	fmt.Printf("Top-K metrics (K=%d, rel>=%.1f):  Precision@K=%.4f  Recall@K=%.4f  NDCG@K=%.4f  HitRate@K=%.4f\n",
		kMetrics, relTh, precK, recK, ndcgK, hitRateK)
	fmt.Printf("Times: load_ratings=%s  load_sim=%s  load_means=%s  split=%s  predict=%s  TOTAL=%s\n",
		tLoadRatings, tLoadSim, tLoadMeans, tSplit, tPredict, tTotal)
	fmt.Printf("Throughput: %.0f preds/s (k_eval=%d)\n", throughput, kEval)

	// -------------------------------------------------------------------------
	// 8) Reporte
	// -------------------------------------------------------------------------
	rep := fmt.Sprintf(
		`== RECOMMEND + EVAL (%s) ==
Sim CSV          : %s
Ratings CSV      : %s
User means       : %v
test_ratio       : %.2f
k_eval           : %d
k_metrics        : %d
rel_threshold    : %.2f
centered (item)  : %v

Evaluated pairs  : %d
MAE              : %.4f
RMSE             : %.4f

Top-K metrics (por usuario):
  Precision@K    : %.4f
  Recall@K       : %.4f
  NDCG@K         : %.4f
  HitRate@K      : %.4f

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
		testRatio, kEval, kMetrics, relTh, centered,
		n, mae, rmse,
		precK, recK, ndcgK, hitRateK,
		throughput,
		tLoadRatings, tLoadSim, tLoadMeans, tSplit, tPredict, tTotal,
	)

	_ = os.WriteFile(reportPath, []byte(rep), 0o644)
	fmt.Printf("Reporte -> %s\n", reportPath)
}

// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------

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

// computeTopKMetrics calcula Precision@K, Recall@K, NDCG@K y HitRate@K
// promediando sobre usuarios.
func computeTopKMetrics(evalByUser map[int][]evalRec, k int, relTh float64) (precK, recK, ndcgK, hitRateK float64) {
	if k <= 0 {
		return 0, 0, 0, 0
	}

	var sumPrec, sumRec, sumNDCG float64
	var usersWithRel, usersTotal, usersHit int

	for _, lst := range evalByUser {
		if len(lst) == 0 {
			continue
		}
		usersTotal++

		// contar relevantes totales
		totalRel := 0
		for _, e := range lst {
			if e.rTrue >= relTh {
				totalRel++
			}
		}
		if totalRel == 0 {
			continue
		}
		usersWithRel++

		// ordenar por predicción descendente
		sort.Slice(lst, func(i, j int) bool { return lst[i].rPred > lst[j].rPred })

		kEff := k
		if len(lst) < kEff {
			kEff = len(lst)
		}

		relInTop := 0
		dcg := 0.0
		for rank := 0; rank < kEff; rank++ {
			if lst[rank].rTrue >= relTh {
				relInTop++
				gain := 1.0
				den := math.Log2(float64(rank) + 2.0) // log2(rank+2)
				dcg += gain / den
			}
		}

		if relInTop > 0 {
			usersHit++
		}

		prec := float64(relInTop) / float64(kEff)
		rec := float64(relInTop) / float64(totalRel)

		// IDCG
		maxRank := kEff
		if totalRel < maxRank {
			maxRank = totalRel
		}
		idcg := 0.0
		for rank := 0; rank < maxRank; rank++ {
			idcg += 1.0 / math.Log2(float64(rank)+2.0)
		}
		ndcg := 0.0
		if idcg > 0 {
			ndcg = dcg / idcg
		}

		sumPrec += prec
		sumRec += rec
		sumNDCG += ndcg
	}

	if usersWithRel > 0 {
		precK = sumPrec / float64(usersWithRel)
		recK = sumRec / float64(usersWithRel)
		ndcgK = sumNDCG / float64(usersWithRel)
	}
	if usersTotal > 0 {
		hitRateK = float64(usersHit) / float64(usersTotal)
	}
	return
}
