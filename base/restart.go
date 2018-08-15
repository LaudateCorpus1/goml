package base

import "math"

//http://ruder.io/deep-learning-optimization-2017/index.html#tuningthelearningrate
func LearningRate(min, max float64, cur_iteration, next_restart int) float64 {

	if min == max {
		return min
	}

	iter_perc := float64(cur_iteration) / float64(next_restart)
	return min + (0.5*(max-min))*(1+math.Cos(iter_perc*math.Pi))
}

type CyclicalLearningDriver struct {
	Min, Max                         float64
	Examples                         int
	Iter, RestartIter, LearningEpoch int
}

func NewCyclicalLearningDriver(Min, Max float64, Examples int) *CyclicalLearningDriver {
	return &CyclicalLearningDriver{
		Min:           Min,
		Max:           Max,
		Examples:      Examples,
		RestartIter:   Examples - 1, //(2^0*Examples)-1
		Iter:          0,
		LearningEpoch: 0,
	}
}

func (dr *CyclicalLearningDriver) Next() float64 {
	lr := LearningRate(dr.Min, dr.Max, dr.Iter, dr.RestartIter)
	if dr.Iter < dr.RestartIter {
		dr.Iter++
	} else {
		dr.LearningEpoch++
		dr.Iter = 0
		dr.RestartIter = (int(math.Pow(2, float64(dr.LearningEpoch))) * dr.Examples) - 1
	}
	return lr
}
