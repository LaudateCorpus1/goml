package base

import (
	"testing"
)

func TestLearningRate(t *testing.T) {
	type args struct {
		min          float64
		max          float64
		cur_epoch    int
		next_restart int
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			args: args{
				min:          .001,
				max:          .1,
				cur_epoch:    0,
				next_restart: 2,
			},
			want: .1,
		},
		{
			args: args{
				min:          .001,
				max:          .1,
				cur_epoch:    1,
				next_restart: 2,
			},
			want: 0.0505,
		},
		{
			args: args{
				min:          .001,
				max:          .1,
				cur_epoch:    2,
				next_restart: 2,
			},
			want: .001,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := LearningRate(tt.args.min, tt.args.max, tt.args.cur_epoch, tt.args.next_restart); got != tt.want {
				t.Errorf("LearningRate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDriver_Next(t *testing.T) {
	type fields struct {
		Min      float64
		Max      float64
		Examples int
	}
	tests := []struct {
		name   string
		fields fields
		want   []float64
	}{
		{
			fields: fields{
				Min:      0,
				Max:      1,
				Examples: 2,
			},
			want: []float64{
				1, 0, 1, .75, LearningRate(0, 1, 2, 3), 0, 1, LearningRate(0, 1, 1, 7),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dr := NewCyclicalLearningDriver(tt.fields.Min, tt.fields.Max, tt.fields.Examples)
			for _, want := range tt.want {
				if got := dr.Next(); got != want {
					t.Errorf("Driver.Next0() = %v, want %v", got, want)
				}
			}
		})
	}
}
