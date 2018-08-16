package linear

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"

	"github.com/bountylabs/goml/base"
	"sync"
	"runtime"
)

// SparseLogistic represents the logistic classification
// model with a sigmoidal hypothesis
//
// https://en.wikipedia.org/wiki/Logistic_regression
//
// The model is currently optimized using Gradient
// Ascent, not Newton's method, etc.
//
// The model expects all expected results in the
// []float64 to come as either a 0 or a 1, and
// will predict the probability that, based on inputs
// x, whether y is 1
type SparseLogistic struct {
	// alpha and maxIterations are used only for
	// GradientDescent during learning. If maxIterations
	// is 0, then GradientDescent will run until the
	// algorithm detects convergance.
	//
	// regularization is used as the regularization
	// term to avoid overfitting within regression.
	// Having a regularization term of 0 is like having
	// _no_ data regularization. The higher the term,
	// the greater the bias on the regression
	alpha          float64
	alphaMax       float64
	regularization float64
	maxIterations  int
	rt             base.RegularizationType

	// method is the optimization method used when training
	// the model
	method base.OptimizationMethod

	// trainingSet and expectedResults are the
	// 'x', and 'y' of the data, expressed as
	// vectors, that the model can optimize from
	trainingSet      []map[int]float64
	expectedResults  []float64

	Parameters []float64 `json:"theta"`

	// Output is the io.Writer used for logging
	// and printing. Defaults to os.Stdout.
	Output io.Writer
}

// NewSparseLogistic takes in a learning rate alpha, a regularization
// parameter value (0 means no regularization, higher value
// means higher bias on the model,) the maximum number of
// iterations the data can go through in gradient descent,
// as well as a training set and expected results for that
// training set.
//
// if you're passing in no training set directly because you want
// to learn using the online method then just declare the number of
// features (it's an integer) as an extra arg after the rest
// of the arguments
//
// DATA FORMAT:
// The SparseLogistic model expects expected results to be either a 0
// or a 1. Predict returns the probability that the item inputted
// is a 1. Obviously this means that the probability that the inputted
// x is a 0 is 1-TheGuess
//
// Example Binary Logistic Regression (Batch GA):
//
//     // optimization method: Batch Gradient Ascent
//     // Learning rate: 1e-4
//     // Regulatization term: 6
//     // Max Iterations: 800
//     // Dataset to learn fron: testX
//     // Expected results dataset: testY
//     model := NewSparseLogistic(base.BatchGD, 1e-4, 6, 800, testX, testY)
//
//     err := model.Learn()
//     if err != nil {
//         panic("SOME ERROR!! RUN!")
//     }
//
//     // now I want to predict off of this
//     // Ordinary Least Squares model!
//     guess, err = model.Predict([]float64{10000,6})
//     if err != nil {
//         panic("AAAARGGGH! SHIVER ME TIMBERS! THESE ROTTEN SCOUNDRELS FOUND AN ERROR!!!")
//     }
func NewSparseLogistic(method base.OptimizationMethod, alpha, alphaMax, regularization float64, rt base.RegularizationType, maxIterations int, trainingSet []map[int]float64, expectedResults []float64, numberOfFeatures int) *SparseLogistic {

	return &SparseLogistic{
		alpha:          alpha,
		alphaMax:       alphaMax,
		regularization: regularization,
		rt:             rt,
		maxIterations:  maxIterations,

		method: method,

		trainingSet:      trainingSet,
		expectedResults:  expectedResults,

		// initialize θ as the zero vector (that is,
		// the vector of all zeros)
		Parameters: make([]float64, numberOfFeatures+1),

		Output: os.Stdout,
	}
}

// UpdateTrainingSet takes in a new training set (variable x)
// as well as a new result set (y). This could be useful if
// you want to retrain a model starting with the parameter
// vector of a previous training session, but most of the time
// wouldn't be used.
func (l *SparseLogistic) UpdateTrainingSet(trainingSet []map[int]float64, expectedResults []float64) error {
	if len(trainingSet) == 0 {
		return fmt.Errorf("Error: length of given training set is 0! Need data!")
	}
	if len(expectedResults) == 0 {
		return fmt.Errorf("Error: length of given result data set is 0! Need expected results!")
	}

	l.trainingSet = trainingSet
	l.expectedResults = expectedResults

	return nil
}

// UpdateLearningRate set's the learning rate of the model
// to the given float64.
func (l *SparseLogistic) UpdateLearningRate(a float64) {
	l.alpha = a
}

// LearningRate returns the learning rate α for gradient
// descent to optimize the model. Could vary as a function
// of something else later, potentially.
func (l *SparseLogistic) LearningRate() float64 {
	return l.alpha
}

// LearningRate returns the learning rate α for gradient
// descent to optimize the model. Could vary as a function
// of something else later, potentially.
func (l *SparseLogistic) LearningRateMax() float64 {
	return l.alphaMax
}

// Examples returns the number of training examples (m)
// that the model currently is training from.
func (l *SparseLogistic) Examples() int {
	return len(l.trainingSet)
}

// MaxIterations returns the number of maximum iterations
// the model will go through in GradientDescent, in the
// worst case
func (l *SparseLogistic) MaxIterations() int {
	return l.maxIterations
}

func (l *SparseLogistic) TrainingError(i int) (float64, error) {
	return l.expectedResults[i]-l.PredictSparse(l.trainingSet[i]), nil
}

// Predict takes in a variable x (an array of floats,) and
// finds the value of the hypothesis function given the
// current parameter vector θ
//
// if normalize is given as true, then the input will
// first be normalized to unit length. Only use this if
// you trained off of normalized inputs and are feeding
// an un-normalized input
func (l *SparseLogistic) Predict(x []float64, normalize ...bool) ([]float64, error) {

	if len(x)+1 != len(l.Parameters) {
		return nil, fmt.Errorf("Error: Parameter vector should be 1 longer than input vector!\n\tLength of x given: %v\n\tLength of parameters: %v\n", len(x), len(l.Parameters))
	}

	if len(normalize) != 0 && normalize[0] {
		base.NormalizePoint(x)
	}

	// include constant term in sum
	sum := l.Parameters[0]

	for i := range x {
		sum += x[i] * l.Parameters[i+1]
	}

	result := 1 / (1 + math.Exp(-sum))

	return []float64{result}, nil
}

func (l *SparseLogistic) PredictSparse(x map[int]float64, normalize ...bool) float64 {

	if len(normalize) != 0 && normalize[0] {
		base.NormalizeSparsePoint(x)
	}

	// include constant term in sum
	sum := l.Parameters[0]

	for i, v := range x {
		sum += v * l.Parameters[i+1]
	}

	result := 1 / (1 + math.Exp(-sum))
	return result
}

// Learn takes the struct's dataset and expected results and runs
// batch gradient descent on them, optimizing theta so you can
// predict based on those results
func (l *SparseLogistic) Learn(file string) error {
	if l.trainingSet == nil || l.expectedResults == nil {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		fmt.Fprintf(l.Output, err.Error())
		return err
	}

	examples := len(l.trainingSet)
	if examples == 0 || len(l.Parameters) == 0 {
		err := fmt.Errorf("ERROR: Attempting to learn with no training examples!\n")
		fmt.Fprintf(l.Output, err.Error())
		return err
	}
	if len(l.expectedResults) == 0 {
		err := fmt.Errorf("ERROR: Attempting to learn with no expected results! This isn't an unsupervised model!! You'll need to include data before you learn :)\n")
		fmt.Fprintf(l.Output, err.Error())
		return err
	}

	fmt.Fprintf(l.Output, "Training:\n\tModel: SparseLogistic (Binary) Classification\n\tOptimization Method: %v\n\tTraining Examples: %v\n\tFeatures: %v\n\tLearning Rate α: %v-%v\n\tRegularization Parameter λ: %v\n\tRegularization Type: %s\n...\n\n", l.method, examples, len(l.Parameters) - 1, l.alphaMax, l.alpha, l.regularization, l.rt.String())

	var err error
	if l.method == base.BatchGD {
		err = base.GradientDescent(l, file)
	} else if l.method == base.StochasticGD {
		err = base.StochasticGradientDescent(l, file)
	} else {
		err = fmt.Errorf("Chose a training method not implemented for SparseLogistic regression")
	}

	if err != nil {
		fmt.Fprintf(l.Output, "\nERROR: Error while learning –\n\t%v\n\n", err)
		return err
	}

	fmt.Fprintf(l.Output, "Training Completed.\n")
	return nil
}

// OnlineLearn runs similar to using a fixed dataset with
// Stochastic Gradient Descent, but it handles data by
// passing it as a channal, and returns errors through
// a channel, which lets it run responsive to inputted data
// from outside the model itself (like using data from the
// stock market at timed intervals or using realtime data
// about the weather.)
//
// The onUpdate callback is called whenever the parameter
// vector theta is changed, so you are able to persist the
// model with the most up to date vector at all times (you
// could persist to a database within the callback, for
// example.) Don't worry about it taking too long and blocking,
// because the callback is spawned into another goroutine.
//
// NOTE that this function is suggested to run in it's own
// goroutine, or at least is designed as such.
//
// NOTE part 2: You can pass in an empty dataset, so long
// as it's not nil, and start pushing after.
//
// NOTE part 3: each example is only looked at as it goes
// through the channel, so if you want to have each example
// looked at more than once you must manually pass the data
// yourself.
//
// NOTE part 4: the optional parameter 'normalize' will
// , if true, normalize all data streamed through the
// channel to unit length. This will affect the outcome
// of the hypothesis, though it could be favorable if
// your data comes in drastically different scales.
//
// Example Online SparseLogistic Regression:
//
//     // create the channel of data and errors
//     stream := make(chan base.Datapoint, 100)
//     errors := make(chan error)
//
//     // notice how we are adding another integer
//     // to the end of the NewSparseLogistic call. This
//     // tells the model to use that number of features
//     // (4) in leu of finding that from the dataset
//     // like you would with batch/stochastic GD
//     //
//     // Also – the 'base.StochasticGD' doesn't affect
//     // anything. You could put batch.
//     model := NewSparseLogistic(base.StochasticGD, .0001, 0, 0, nil, nil, 4)
//
//     go model.OnlineLearn(errors, stream, func(theta [][]float64) {
//         // do something with the new theta (persist
//         // to database?) in here.
//     })
//
//     go func() {
//         for iterations := 0; iterations < 20; iterations++ {
//             for i := -200.0; abs(i) > 1; i *= -0.75 {
//                 for j := -200.0; abs(j) > 1; j *= -0.75 {
//                     for k := -200.0; abs(k) > 1; k *= -0.75 {
//                         for l := -200.0; abs(l) > 1; l *= -0.75 {
//                             if i/2+2*k-4*j+2*l+3 > 0 {
//                                 stream <- base.Datapoint{
//                                     X: []float64{i, j, k, l},
//                                     Y: []float64{1.0},
//                                 }
//                             } else {
//                                 stream <- base.Datapoint{
//                                     X: []float64{i, j, k, l},
//                                     Y: []float64{0.0},
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//
//         // close the dataset to tell the model
//         // to stop learning when it finishes reading
//         // what's left in the channel
//         close(stream)
//     }()
//
//     // this will block until the error
//     // channel is closed in the learning
//     // function (it will, don't worry!)
//     for {
//         err, more := <-errors
//         if err != nil {
//             panic("THERE WAS AN ERROR!!! RUN!!!!")
//         }
//         if !more {
//             break
//         }
//     }
//
//     // Below here all the learning is completed
//
//     // predict like usual
//     guess, err = model.Predict([]float64{42,6,10,-32})
//     if err != nil {
//         panic("AAAARGGGH! SHIVER ME TIMBERS! THESE ROTTEN SCOUNDRELS FOUND AN ERROR!!!")
//     }
func (l *SparseLogistic) OnlineLearn(errors chan error, dataset chan base.Datapoint, onUpdate func([][]float64), normalize ...bool) {
	if errors == nil {
		errors = make(chan error)
	}
	if dataset == nil {
		errors <- fmt.Errorf("ERROR: Attempting to learn with a nil data stream!\n")
		close(errors)
		return
	}

	fmt.Fprintf(l.Output, "Training:\n\tModel: SparseLogistic (Binary) Classifier\n\tOptimization Method: Online Stochastic Gradient Descent\n\tFeatures: %v\n\tLearning Rate α: %v-%v\n...\n\n", len(l.Parameters), l.alphaMax, l.alpha)

	norm := len(normalize) != 0 && normalize[0]
	var point base.Datapoint
	var more bool

	for {
		point, more = <-dataset

		if more {
			if len(point.Y) != 1 {
				errors <- fmt.Errorf("ERROR: point.Y must have a length of 1. Point: %v", point)
			}

			if norm {
				base.NormalizePoint(point.X)
			}

			newTheta := make([]float64, len(l.Parameters))
			for j := range l.Parameters {

				// find the gradient using the point
				// from the channel (different than
				// calling from the dataset so we need
				// to have a new function instead of calling
				// Dij(i, j))
				dj, err := func(point base.Datapoint, j int) (float64, error) {
					prediction, err := l.Predict(point.X)
					if err != nil {
						return 0, err
					}

					// account for constant term
					// x is x[i][j] via Andrew Ng's terminology
					var x float64
					if j == 0 {
						x = 1
					} else {
						x = point.X[j-1]
					}

					var gradient float64
					gradient = (point.Y[0] - prediction[0]) * x

					// add in the regularization term
					// λ*θ[j]
					//
					// notice that we don't count the
					// constant term
					if j != 0 {
						gradient -= l.Regularization(j)
					}

					return gradient, nil
				}(point, j)
				if err != nil {
					errors <- err
					continue
				}

				newTheta[j] = l.Parameters[j] + l.alpha*dj
			}

			// now simultaneously update Theta
			for j := range l.Parameters {
				newθ := newTheta[j]
				if math.IsInf(newθ, 0) || math.IsNaN(newθ) {
					errors <- fmt.Errorf("Sorry! Learning diverged. Some value of the parameter vector theta is ±Inf or NaN")
					continue
				}
				l.Parameters[j] = newθ
			}

			go onUpdate([][]float64{l.Parameters})

		} else {
			fmt.Fprintf(l.Output, "Training Completed.\n%v\n\n", l)
			close(errors)
			return
		}
	}
}

// String implements the fmt interface for clean printing. Here
// we're using it to print the model as the equation h(θ)=...
// where h is the logistic hypothesis model
func (l *SparseLogistic) String() string {
	features := len(l.Parameters) - 1
	if len(l.Parameters) == 0 {
		fmt.Fprintf(l.Output, "ERROR: Attempting to print model with the 0 vector as it's parameter vector! Train first!\n")
	}
	var buffer bytes.Buffer

	buffer.WriteString(fmt.Sprintf("h(θ,x) = 1 / (1 + exp(-θx))\nθx = %.3f + ", l.Parameters[0]))

	length := features + 1
	for i := 1; i < length; i++ {
		buffer.WriteString(fmt.Sprintf("%.5f(x[%d])", l.Parameters[i], i))

		if i != features {
			buffer.WriteString(fmt.Sprintf(" + "))
		}
	}

	return buffer.String()
}

//Used to speed up Batch Gradient Descent
func (l *SparseLogistic) PredictAll() ([]float64, float64) {

	n_cores := runtime.NumCPU()
	predictions := make([]float64, len(l.trainingSet))
	errors := make([]float64, n_cores)


	wg := &sync.WaitGroup{}
	wg.Add(n_cores)
	for core := 0; core < n_cores; core++ {

		go func(core int) {

			/*
				25 = 101 / 4
				start = 0 * 25, 1 * 25, 2 * 25, 3 * 25
				end = 25, 50, 75, 101
			*/

			nObservationsPerCore := len(l.trainingSet) / n_cores
			start := core * nObservationsPerCore
			end := start + nObservationsPerCore
			if core == n_cores-1 {
				end = len(l.trainingSet)
			}

			for i := start; i < end; i++ {
				predictions[i] = l.PredictSparse(l.trainingSet[i])
				prediction_error := l.expectedResults[i] - predictions[i]
				errors[core] += (prediction_error * prediction_error)
			}
			wg.Done()
		}(core)
	}
	wg.Wait()

	//fan in error results
	var error_sum float64 = 0
	for _, core_err := range errors {
		error_sum += core_err
	}

	return predictions, math.Sqrt(error_sum/float64(len(predictions)))
}

// Dj returns the partial derivative of the cost function J(θ)
// with respect to theta[j] where theta is the parameter vector
// associated with our hypothesis function Predict (upon which
// we are optimizing
func (l *SparseLogistic) Dj(j int, predictions []float64) (float64, error) {
	if j > len(l.Parameters)-1 {
		return 0, fmt.Errorf("J (%v) would index out of the bounds of the training set data (len: %v)", j, len(l.Parameters))
	}

	var sum float64

	for i := range l.trainingSet {
		// account for constant term
		// x is x[i][j] via Andrew Ng's terminology
		var x float64
		if j == 0 {
			x = 1
		} else {
			x = l.trainingSet[i][j-1]
		}

		sum += (l.expectedResults[i]-predictions[i]) * x
	}

	// add in the regularization term
	// λ*θ[j]
	//
	// notice that we don't count the
	// constant term
	if j != 0 {
		sum -= l.Regularization(j)
	}

	return sum, nil
}

// Dij returns the derivative of the cost function
// J(θ) with respect to the j-th parameter of
// the hypothesis, θ[j], for the training example
// x[i]. Used in Stochastic Gradient Descent.
//
// assumes that i,j is within the bounds of the
// data they are looking up! (because this is getting
// called so much, it needs to be efficient with
// comparisons)
func (l *SparseLogistic) Dij(i int, j int, prediction_error float64) float64 {

	// account for constant term
	// x is x[i][j] via Andrew Ng's terminology
	var x float64
	if j == 0 {
		x = 1
	} else {
		x = l.trainingSet[i][j-1]
	}

	var gradient float64
	gradient = prediction_error * x

	// add in the regularization term
	// λ*θ[j]
	//
	// notice that we don't count the
	// constant term
	if j != 0 {
		gradient -= l.Regularization(j)
	}

	return gradient
}

func (l *SparseLogistic) Regularization(j int) float64 {
	switch l.rt {
	case base.L1:
		return l.regularization * NormAbs(l.Parameters[j])
	case base.L2:
		return l.regularization * (l.Parameters[j])
	default:
		panic("unkown regularization type")
	}
}

func NormAbs(in float64) float64 {
	switch {
	case in > 0:
		return 1
	case in < 0:
		return -1
	default:
		return 0
	}
}

// Theta returns the parameter vector θ for use in persisting
// the model, and optimizing the model through gradient descent
// ( or other methods like Newton's Method)
func (l *SparseLogistic) Theta() []float64 {
	return l.Parameters
}

// PersistToFile takes in an absolute filepath and saves the
// parameter vector θ to the file, which can be restored later.
// The function will take paths from the current directory, but
// functions
//
// The data is stored as JSON because it's one of the most
// efficient storage method (you only need one comma extra
// per feature + two brackets, total!) And it's extendable.
func (l *SparseLogistic) PersistToFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to persist your model to a file with no path!! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := json.Marshal(l.Parameters)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(path, bytes, os.ModePerm)
	if err != nil {
		return err
	}

	return nil
}

// RestoreFromFile takes in a path to a parameter vector theta
// and assigns the model it's operating on's parameter vector
// to that.
//
// The path must ba an absolute path or a path from the current
// directory
//
// This would be useful in persisting data between running
// a model on data, or for graphing a dataset with a fit in
// another framework like Julia/Gadfly.
func (l *SparseLogistic) RestoreFromFile(path string) error {
	if path == "" {
		return fmt.Errorf("ERROR: you just tried to restore your model from a file with no path! That's a no-no. Try it with a valid filepath")
	}

	bytes, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	err = json.Unmarshal(bytes, &l.Parameters)
	if err != nil {
		return err
	}

	return nil
}
