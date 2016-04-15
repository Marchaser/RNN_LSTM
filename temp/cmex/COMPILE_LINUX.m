function COMPILE
% D:\blitz-0.10\src\globals.cpp 
mex CXX=icc lstm_mex.cpp -DBZ_THREADSAFE -DUSE_OMP -I"/home/wl239/CRoutines" OPTIMFLAGS="-O3 -DNDEBUG -fno-alias" CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp -mkl"
% mex ../Iter1Mex.cpp -DBZ_THREADSAFE -DUSE_OMP -I"/home/wl239/CRoutines" OPTIMFLAGS="-O3 -DNDEBUG -fno-alias" CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"

% copyfile('Iter1Mex.mexa64','v1/');
% mex -g ..\Iter1Mex.cpp D:\blitz-0.10\src\globals.cpp -I"D:\blitz-0.10" -I"D:\cpp_local\routines" -I"C:\Program Files (x86)\Intel\Composer XE\mkl\include"
% -DBZ_DEBUG 
end
