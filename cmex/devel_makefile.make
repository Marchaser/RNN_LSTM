MKLROOT = /share/apps/intel/composer_xe_2015.3.187/mkl/lib/intel64
MATLABROOT = /share/apps/matlab/R2014b
net:
	icpc lstmNet.cpp -O2 -DUSE_OMP \
		-DTYPENAME_DOUBLE -o lstmNet_double.mexa64 \
		-Wl,--start-group $(MKLROOT)/libmkl_intel_lp64.a \
		$(MKLROOT)/libmkl_core.a \
		$(MKLROOT)/libmkl_intel_thread.a \
		-Wl,--end-group -lpthread -lm -ldl \
		-I"/home/wl239/CRoutines" \
		-I$(MATLABROOT)/extern/include -I$(MATLABROOT)/simulink/include -DMATLAB_MEX_FILE -ansi -D_GNU_SOURCE -fexceptions -fPIC -fno-omit-frame-pointer -pthread -shared -m64 -DNOCHECK -DMX_COMPAT_32 -DNDEBUG -Wl,--version-script,$(MATLABROOT)/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined -Wl,-rpath-link,$(MATLABROOT)/bin/glnxa64 -L$(MATLABROOT)/bin/glnxa64 -lmx -ldl -lmex -lmat -lstdc++ -qopenmp
	icpc lstmNet.cpp -O2 -DUSE_OMP \
		-DTYPENAME_FLOAT -o lstmNet_single.mexa64 \
		-Wl,--start-group $(MKLROOT)/libmkl_intel_lp64.a \
		$(MKLROOT)/libmkl_core.a \
		$(MKLROOT)/libmkl_intel_thread.a \
		-Wl,--end-group -lpthread -lm -ldl \
		-I"/home/wl239/CRoutines" \
		-I$(MATLABROOT)/extern/include -I$(MATLABROOT)/simulink/include -DMATLAB_MEX_FILE -ansi -D_GNU_SOURCE -fexceptions -fPIC -fno-omit-frame-pointer -pthread -shared -m64 -DNOCHECK -DMX_COMPAT_32 -DNDEBUG -Wl,--version-script,$(MATLABROOT)/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined -Wl,-rpath-link,$(MATLABROOT)/bin/glnxa64 -L$(MATLABROOT)/bin/glnxa64 -lmx -ldl -lmex -lmat -lstdc++ -qopenmp
