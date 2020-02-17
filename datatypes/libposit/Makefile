libposit-wrapper: libposit-wrapper.cc
				cd libposit && ./configure && make
				$(CXX) -Ilibposit/generated -lmpfr -lgmp --std=c++14 -shared -o libposit-wrapper.so -fPIC libposit-wrapper.cc libposit/libposit.a
