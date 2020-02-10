libposit-wrapper: libposit-wrapper.cc
				cd libposit && ./configure && make
				$(CXX) -Ilibposit/generated -lmpfr -lgmp -lposit -Llibposit --std=c++14 -shared -o libposit-wrapper.so -fPIC libposit-wrapper.cc
