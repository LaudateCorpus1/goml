package base

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"
	"runtime"
)

// GradientDescent operates on a Descendable model and
// further optimizes the parameter vector Theta of the
// model, which is then used within the Predict function.
//
// Gradient Descent follows the following algorithm:
// θ[j] := θ[j] + α·∇J(θ)
//
// where J(θ) is the cost function, α is the learning
// rate, and θ[j] is the j-th value in the parameter
// vector
func GradientDescent(d Descendable, file string) error {
	Theta := d.Theta()
	Alpha := d.LearningRate()
	MaxIterations := d.MaxIterations()

	// if the iterations given is 0, set it to be
	// 250 (seems reasonable base value)
	if MaxIterations == 0 {
		MaxIterations = 250
	}

	previous_rmse := -1.0


	// Stop iterating if the number of iterations exceeds
	// the limit
	for iter := 0; iter < MaxIterations; iter++ {


		start := time.Now()

		predictions, rmse := d.PredictAll()
		if math.Abs(rmse - previous_rmse) < 1e-6 {
			fmt.Println("Convergence delta=", rmse-previous_rmse)
			break
		} else {
			previous_rmse = rmse
		}

		newTheta, err := BatchNewThetaParallel(Theta, d, Alpha, predictions)
		if err != nil {
			return err
		}

		// now simultaneously update Theta
		copy(Theta, newTheta)

		fmt.Println(iter, "ttd:", time.Now().Sub(start)*time.Duration(MaxIterations-(iter + 1)), rmse)
		if file != "" {
			d.PersistToFile(file)
		}

	}

	return nil
}


func BatchNewTheta(Theta []float64, d Descendable, Alpha float64, predictions []float64) ([]float64, error) {
	newTheta := make([]float64, len(Theta))
	for j := range Theta {
		dj, err := d.Dj(j, predictions)
		if err != nil {
			return nil, err
		}
		newTheta[j] = Theta[j] - Alpha*dj
	}
	return newTheta, nil
}

func BatchNewThetaParallel(Theta []float64, d Descendable, Alpha float64, predictions []float64) ([]float64, error) {

	newTheta := make([]float64, len(Theta))
	n_cores := runtime.NumCPU()

	if len(Theta) < n_cores {
		n_cores = len(Theta)
	}

	//nObservationsPerCore is always >= 1
	nFeaturesPerCore := int(math.Ceil(float64(len(Theta)) / float64(n_cores)))
	wg := &sync.WaitGroup{}
	wg.Add(n_cores)

	for core := 0; core < n_cores; core++ {

		go func(core int) {

			/*
				25 = 101 / 4
				start = 0 * 25, 1 * 25, 2 * 25, 3 * 25
				end = 25, 50, 75, 101
			*/

			start := core * nFeaturesPerCore
			end := start + nFeaturesPerCore
			if end > len(Theta) {
				end = len(Theta)
			}

			for j := start; j < end; j++ {
				dj, err := d.Dj(j, predictions)
				if err != nil {
					panic(err)
				}
				newTheta[j] = Theta[j] - Alpha*dj
			}
			wg.Done()
		}(core)
	}
	wg.Wait()
	return newTheta, nil
}


// StochasticGradientDescent operates on a StochasticDescendable
// model and further optimizes the parameter vector Theta of the
// model, which is then used within the Predict function.
// Stochastic gradient descent updates the parameter vector
// after looking at each individual training example, which
// can result in never converging to the absolute minimum; even
// raising the cost function potentially, but it will typically
// converge faster than batch gradient descent (implemented as
// func GradientDescent(d Descendable) error) because of that very
// difference.
//
// Gradient Descent follows the following algorithm:
// θ[j] := θ[j] + α·∇J(θ)
//
// where J(θ) is the cost function, α is the learning
// rate, and θ[j] is the j-th value in the parameter
// vector
func StochasticGradientDescent(d StochasticDescendable, file string) error {

	var (
		Theta          = d.Theta()
		MaxIterations  = d.MaxIterations()
		Examples       = d.Examples()
		LearningDriver = NewCyclicalLearningDriver(d.LearningRate(), d.LearningRateMax(), Examples)
	)

	//Create an array of training indices
	r := rand.New(rand.NewSource(2))
	indices := make([]int, Examples, Examples)
	for i := 0; i < Examples; i++ {
		indices[i] = i
	}

	// if the iterations given is 0, set it to be
	// 250 (seems reasonable base value)
	if MaxIterations == 0 {
		MaxIterations = 250
	}

	n_features := len(Theta)
	previous_rmse := -1.0

	// Stop iterating if the number of iterations exceeds
	// the limit
	for iter := 0; iter < MaxIterations; iter++ {

		newTheta := make([]float64, n_features)

		var error_sum float64 = 0
		start := time.Now()
		shuffle(r, indices)

		for trainingIteration := 0; trainingIteration < Examples; trainingIteration++ {

			i := indices[trainingIteration]

			prediction_error, err := d.TrainingError(i)
			if err != nil {
				return err
			}

			error_sum += (prediction_error * prediction_error)

			if len(Theta) > 10000 {
				NewThetaParallel(Theta, d, i, prediction_error, LearningDriver.Next(), newTheta)
			} else {
				NewTheta(Theta, d, i, prediction_error, LearningDriver.Next(), newTheta)
			}

			copy(Theta, newTheta)


			//if trainingIteration % 1000 == 0 {
			//	fmt.Println(iter, "Sqrt(Err^2/N)", math.Sqrt(error_sum/float64(trainingIteration)))
			//}

		}

		rmse := math.Sqrt(error_sum/float64(Examples))
		if math.Abs(rmse - previous_rmse) < 1e-6 {
			fmt.Println("Convergence delta=", rmse-previous_rmse)
			break
		} else {
			previous_rmse = rmse
		}

		fmt.Println(iter, "ttd:", time.Now().Sub(start)*time.Duration(MaxIterations-(iter + 1)), rmse)

		if file != "" {
			d.PersistToFile(file)
		}

	}

	return nil
}

func shuffle(r *rand.Rand, x []int) {
	for i := range x {
		j := r.Intn(i + 1)
		x[i], x[j] = x[j], x[i]
	}
}

func NewThetaParallel(Theta []float64, d StochasticDescendable, i int, prediction_error float64, Alpha float64, newTheta []float64) {

	n_cores := runtime.NumCPU()
	wg := &sync.WaitGroup{}
	wg.Add(n_cores)
	for core := 0; core < n_cores; core++ {

		go func(core int) {

			/*
				25 = 101 / 4
				start = 0 * 25, 1 * 25, 2 * 25, 3 * 25
				end = 25, 50, 75, 101
			*/

			n_samples := len(Theta) / n_cores
			start := core * n_samples
			end := start + n_samples
			if core == n_cores-1 {
				end = len(Theta)
			}

			for j := start; j < end; j++ {
				dj := d.Dij(i, j, prediction_error)
				newθ := Theta[j] - (Alpha * dj)
				if math.IsInf(newθ, 0) || math.IsNaN(newθ) {
					log.Fatalf("Sorry! Learning diverged. Some value of the parameter vector(%d) theta is ±Inf(%v) or NaN(%v)", j, math.IsInf(newθ, 0), math.IsNaN(newθ))
				}
				newTheta[j] = newθ
			}
			wg.Done()
		}(core)
	}
	wg.Wait()
}

func NewTheta(Theta []float64, d StochasticDescendable, i int, prediction_error float64, Alpha float64, newTheta []float64) {

	for j := range Theta {
		dj := d.Dij(i, j, prediction_error)
		newThetaJ := Theta[j] - (Alpha * dj)
		if math.IsInf(newThetaJ, 0) || math.IsNaN(newThetaJ) {
			log.Fatalf("Sorry! Learning diverged. Some value of the parameter vector(%d) theta is ±Inf(%v) or NaN(%v)", j, math.IsInf(newThetaJ, 0), math.IsNaN(newThetaJ))
		}
		newTheta[j] = newThetaJ
	}

}
