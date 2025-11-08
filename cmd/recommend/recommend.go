//go:build recommend
// +build recommend

package main

/*
RECOMMEND + EVALUATION (secuencial)
- Genera un hold-out 10% por usuario para test (en memoria).
- Predice con:
    * user-based + Pearson (usa user_topk_pearson.csv y user_means.csv)
    * item-based + Coseno  (usa item_topk_cosine.csv)
- Calcula MAE y RMSE.

Entradas:
  - artifacts/ratings_ui.csv          (uIdx,iIdx,rating)
  - artifacts/sim/user_topk_pearson.csv   o   artifacts/sim/item_topk_cosine.csv
  - artifacts/user_means.csv          (solo modelo user-based)

Parámetros:
  --model=user|item
  --sim=path/to/sim.csv
  --test_ratio=0.1
  --k_eval=10   (opcional: límite de vecinos a usar en predicción; por defecto usa todos del CSV)

Salida (consola):
  - MAE, RMSE y conteo de pares evaluados
*/

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
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

// 1) Cargar ratings por usuario y por ítem
type ur struct {
	i int
	r float64
}
type ir struct {
	u int
	r float64
}

func main() {
	var model string
	var simPath string
	var testRatio float64
	var kEval int
	flag.StringVar(&model, "model", "user", "user | item")
	flag.StringVar(&simPath, "sim", "", "ruta del CSV de similitud")
	flag.Float64Var(&testRatio, "test_ratio", 0.1, "proporción de test por usuario")
	flag.IntVar(&kEval, "k_eval", 0, "si >0, usa como límite de vecinos al predecir")
	flag.Parse()
	if simPath == "" {
		panic("--sim requerido (ruta a user_topk_pearson.csv o item_topk_cosine.csv)")
	}

	users := make(map[int][]ur) // u -> [(i,r)]
	items := make(map[int][]ir) // i -> [(u,r)]

	f, _ := os.Open(tripletsPath)
	rd := csv.NewReader(bufio.NewReader(f))
	_, _ = rd.Read() // header
	maxU, maxI := 0, 0
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
		if u+1 > maxU {
			maxU = u + 1
		}
		if i+1 > maxI {
			maxI = i + 1
		}
	}
	f.Close()

	// 2) Cargar similitudes
	sim := make(map[int][]edge) // nodo -> vecinos ordenados (como fueron escritos)
	sf, _ := os.Open(simPath)
	sr := csv.NewReader(bufio.NewReader(sf))
	_, _ = sr.Read()
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

	// 3) Medias de usuario (si modelo user-based)
	means := make(map[int]float64)
	if model == "user" {
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
	}

	// 4) Split hold-out por usuario
	rand.Seed(time.Now().UnixNano())
	type testPair struct {
		u, i int
		r    float64
	}
	var test []testPair
	train := make(map[int]map[int]float64) // u -> i -> r
	for u, lst := range users {
		perm := rand.Perm(len(lst))
		szTest := int(math.Max(1, math.Round(testRatio*float64(len(lst)))))
		if szTest > len(lst)-1 { // deja al menos 1 en train
			szTest = len(lst) - 1
		}
		if szTest < 1 {
			szTest = 1
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

	// 5) Predicción y métricas
	var absSum, sqSum float64
	var n int

	for _, t := range test {
		var pred float64
		if model == "user" {
			// vecinos de u que hayan valorado i
			nu := sim[t.u]
			if kEval > 0 && len(nu) > kEval {
				nu = nu[:kEval]
			}
			var num, den float64
			for _, e := range nu {
				v := e.to
				// buscar rating de v sobre i
				rv := ratingFromList(items[t.i], v) // búsqueda lineal sobre item->[(u,r)]
				if rv <= 0 {                        // no valoró (MovieLens ratings >= 0.5)
					continue
				}
				num += e.w * (rv - means[v])
				den += math.Abs(e.w)
			}
			if den == 0 {
				pred = means[t.u] // fallback: media del usuario
			} else {
				pred = means[t.u] + num/den
			}
			pred = clamp(pred, 0.5, 5.0)
		} else {
			// item-based: vecinos del ítem i
			ni := sim[t.i]
			if kEval > 0 && len(ni) > kEval {
				ni = ni[:kEval]
			}
			var num, den float64
			// ítems que u ha valorado
			uj := train[t.u]
			for _, e := range ni {
				if rj, ok := uj[e.to]; ok {
					num += e.w * rj
					den += math.Abs(e.w)
				}
			}
			if den == 0 {
				// fallback: media de los ratings del usuario en train
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

	mae := absSum / float64(n)
	rmse := math.Sqrt(sqSum / float64(n))
	fmt.Printf("[MODEL=%s] eval=%d  MAE=%.4f  RMSE=%.4f\n", strings.ToUpper(model), n, mae, rmse)
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
