all: execute

execute: 
	g++ -std=c++17 mtx_features.cc -lsparsebase -fopenmp -lgomp -std=c++17 -o reorder

clean:
	rm -f reorder
