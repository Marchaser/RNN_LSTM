#pragma once

#include "mkl.h"
#include "string.h"
#include "math.h"

#ifdef USE_OMP
#include <omp.h>
#endif

#define SIGMOID(x) 1/(1+exp(-x))
#define TANH(x) (1-exp(-2*x)/(1+exp(-2*x)))

#define RNN_ERR_ALLOC_BATCH_SIZE_TOO_SMALL -1
#define RNN_ERR_WEIGHTS_UNINITIATED -2
#define RNN_ERR_DWEIGHTS_UNINITIATED -3

#pragma region LstmLayer

template <typename T>
class LstmLayer {
private:
	T* memory;

public:
	int xDim;
	int hDim;
	int periods;
	int xhDim;
	int gifoDim;

	int batchSize;
	int memorySize;

	int hasPreallocSpace;
	int isInit;
	
	// Constructor
	LstmLayer() : xDim(0), hDim(0), periods(0), batchSize(0), gifoDim(0), xhDim(0), memorySize(0), hasPreallocSpace(0), isInit(0) {};
	LstmLayer(int _xDim, int _hDim, int _periods, int _batchSize, T* _prealloc_space = 0)
		: xDim(_xDim), hDim(_hDim), periods(_periods), batchSize(_batchSize), memory(_prealloc_space), isInit(0)
	{
		hasPreallocSpace = (_prealloc_space != 0);
		init();
	}

	void init()
	{
		xhDim = xDim + hDim;
		gifoDim = hDim * 4;
		memorySize = computeMemorySize();

		if (memory != 0)
		{
			assign_memory();
		}
		else
		{
			alloc();
		}

		// Set constant
		for (int j = 0; j < batchSize; j++)
		{
			ONES_BATCH_SIZE[j] = 1;
		}
		memset(ZEROS_H_STRIDE, 0, sizeof(T)*hDim*batchSize);

		isInit = 1;
	}

	// Destructor
	~LstmLayer() 
	{
		dealloc();
	}

	T* weights_x;
	T* weights_h;
	T* biases;

	// Used for forward propagation
	T* x_t;
	T* gifo_lin_t;
	T* g_lin;
	T* i_lin;
	T* f_lin;
	T* o_lin;
	T* g_t;
	T* i_t;
	T* f_t;
	T* o_t;
	T* h_t;
	T* s_t;
	T* tanhs_t;

	// Used for backward propagation
	// Information from the upper layer
	T* dtop_t;
	// Information within current period
	T* dh;
	T* ds;
	T* doo;
	T* dtanhs;
	T* dg;
	T* di;
	T* df;
	T* dgifo_lin;
	// Information down to time period
	T* dhm;
	T* dsm;
	// Information down to the lower layer
	T* dx_t;

	// Derivative of weights
	T* dweights_x;
	T* dweights_h;
	T* dbiases;

	// Temp space used for computation
	T* g_temp;
	// Constants used for computation
	T* ONES_BATCH_SIZE;
	T* ZEROS_H_STRIDE;

	void init_dweights()
	{
		memset(dweights_x, 0, sizeof(T)*gifoDim*xDim);
		memset(dweights_h, 0, sizeof(T)*gifoDim*hDim);
		memset(dbiases, 0, sizeof(T)*gifoDim);
	}

	void dealloc();
	void alloc();
	int computeMemorySize();
	void assign_memory();

	int forward_pass(int t);
	int back_propagation(int t);
};

template<typename T>
int LstmLayer<T>::computeMemorySize()
{
	int ptr = 0;

	// dweights_x = ptr;
	ptr += xDim*gifoDim;
	// dweights_h = ptr;
	ptr += hDim*gifoDim;
	// dbiases = ptr;
	ptr += gifoDim;

	// x_t = ptr;
	ptr += xDim*batchSize*periods;

	// gifo_lin_t = ptr;
	ptr += gifoDim*batchSize*periods;

	// g_lin = ptr;
	ptr += hDim*batchSize;
	// i_lin = ptr;
	ptr += hDim*batchSize;
	// f_lin = ptr;
	ptr += hDim*batchSize;
	// o_lin = ptr;
	ptr += hDim*batchSize;

	// g_t = ptr;
	ptr += hDim*batchSize*periods;
	// i_t = ptr;
	ptr += hDim*batchSize*periods;
	// f_t = ptr;
	ptr += hDim*batchSize*periods;
	// o_t = ptr;
	ptr += hDim*batchSize*periods;
	// h_t = ptr;
	ptr += hDim*batchSize*periods;
	// s_t = ptr;
	ptr += hDim*batchSize*periods;
	// tanhs_t = ptr;
	ptr += hDim*batchSize*periods;

	// Used for backward propagation
	// Information from the upper layer
	// dtop_t = ptr;
	ptr += hDim*batchSize*periods;
	// Information within current period
	// dh = ptr;
	ptr += hDim*batchSize;
	// ds = ptr;
	ptr += hDim*batchSize;
	// doo = ptr;
	ptr += hDim*batchSize;
	// dtanhs = ptr;
	ptr += hDim*batchSize;
	// dg = ptr;
	ptr += hDim*batchSize;
	// di = ptr;
	ptr += hDim*batchSize;
	// df = ptr;
	ptr += hDim*batchSize;
	// dgifo_lin = ptr;
	ptr += gifoDim*batchSize;
	// Information down to time period
	// dhm = ptr;
	ptr += hDim*batchSize;
	// dsm = ptr;
	ptr += hDim*batchSize;
	// Information down to the lower layer
	// dx_t = ptr;
	ptr += xDim*batchSize*periods;

	// Temporary and constant space
	// g_temp = ptr;
	ptr += hDim*batchSize;

	// ONES_BATCH_SIZE = ptr;
	ptr += batchSize;

	// ZEROS_H_STRIDE = ptr;
	ptr += hDim*batchSize;

	return ptr;
}

template<typename T>
void LstmLayer<T>::assign_memory()
{
	T* ptr = memory;

	dweights_x = ptr;
	ptr += xDim*gifoDim;
	dweights_h = ptr;
	ptr += hDim*gifoDim;
	dbiases = ptr;
	ptr += gifoDim;

	x_t = ptr;
	ptr += xDim*batchSize*periods;

	gifo_lin_t = ptr;
	ptr += gifoDim*batchSize*periods;

	g_lin = ptr;
	ptr += hDim*batchSize;
	i_lin = ptr;
	ptr += hDim*batchSize;
	f_lin = ptr;
	ptr += hDim*batchSize;
	o_lin = ptr;
	ptr += hDim*batchSize;

	g_t = ptr;
	ptr += hDim*batchSize*periods;
	i_t = ptr;
	ptr += hDim*batchSize*periods;
	f_t = ptr;
	ptr += hDim*batchSize*periods;
	o_t = ptr;
	ptr += hDim*batchSize*periods;
	h_t = ptr;
	ptr += hDim*batchSize*periods;
	s_t = ptr;
	ptr += hDim*batchSize*periods;
	tanhs_t = ptr;
	ptr += hDim*batchSize*periods;

	// Used for backward propagation
	// Information from the upper layer
	dtop_t = ptr;
	ptr += hDim*batchSize*periods;
	// Information within current period
	dh = ptr;
	ptr += hDim*batchSize;
	ds = ptr;
	ptr += hDim*batchSize;
	doo = ptr;
	ptr += hDim*batchSize;
	dtanhs = ptr;
	ptr += hDim*batchSize;
	dg = ptr;
	ptr += hDim*batchSize;
	di = ptr;
	ptr += hDim*batchSize;
	df = ptr;
	ptr += hDim*batchSize;
	dgifo_lin = ptr;
	ptr += gifoDim*batchSize;
	// Information down to time period
	dhm = ptr;
	ptr += hDim*batchSize;
	dsm = ptr;
	ptr += hDim*batchSize;
	// Information down to the lower layer
	dx_t = ptr;
	ptr += xDim*batchSize*periods;

	// Temporary and constant space
	g_temp = ptr;
	ptr += hDim*batchSize;

	ONES_BATCH_SIZE = ptr;
	ptr += batchSize;

	ZEROS_H_STRIDE = ptr;
	ptr += hDim*batchSize;
}

template<typename T>
void LstmLayer<T>::dealloc()
{
	if (hasPreallocSpace)
		return;

	// Deallocate space
	free(x_t);

	free(gifo_lin_t);

	free(g_lin);
	free(i_lin);
	free(f_lin);
	free(o_lin);

	free(g_t);
	free(i_t);
	free(f_t);
	free(o_t);
	free(h_t);
	free(s_t);
	free(tanhs_t);

	// Used for backward propagation
	// Information from the upper layer
	free(dtop_t);
	// Information within current period
	free(dh);
	free(ds);
	free(doo);
	free(dtanhs);
	free(dg);
	free(di);
	free(df);
	free(dgifo_lin);
	// Information down to time period
	free(dhm);
	free(dsm);
	// Information down to the lower layer
	free(dx_t);

	// Temporary and constant space
	free(g_temp);

	free(ONES_BATCH_SIZE);
	free(ZEROS_H_STRIDE);
}

template<typename T>
void LstmLayer<T>::alloc()
{
	if (memory != 0)
		return;

	// Allocate memory
	x_t = (T*)malloc(sizeof(T)*xDim*batchSize*periods);

	gifo_lin_t = (T*)malloc(sizeof(T)*gifoDim*batchSize*periods);

	g_lin = (T*)malloc(sizeof(T)*hDim*batchSize);
	i_lin = (T*)malloc(sizeof(T)*hDim*batchSize);
	f_lin = (T*)malloc(sizeof(T)*hDim*batchSize);
	o_lin = (T*)malloc(sizeof(T)*hDim*batchSize);

	g_t = (T*)malloc(sizeof(T)*hDim*batchSize*periods);
	i_t = (T*)malloc(sizeof(T)*hDim*batchSize*periods);
	f_t = (T*)malloc(sizeof(T)*hDim*batchSize*periods);
	o_t = (T*)malloc(sizeof(T)*hDim*batchSize*periods);
	h_t = (T*)malloc(sizeof(T)*hDim*batchSize*periods);
	s_t = (T*)malloc(sizeof(T)*hDim*batchSize*periods);
	tanhs_t = (T*)malloc(sizeof(T)*hDim*batchSize*periods);

	// Used for backward propagation
	// Information from the upper layer
	dtop_t = (T*)malloc(sizeof(T)*hDim*batchSize*periods);
	// Information within current period
	dh = (T*)malloc(sizeof(T)*hDim*batchSize);
	ds = (T*)malloc(sizeof(T)*hDim*batchSize);
	doo = (T*)malloc(sizeof(T)*hDim*batchSize);
	dtanhs = (T*)malloc(sizeof(T)*hDim*batchSize);
	dg = (T*)malloc(sizeof(T)*hDim*batchSize);
	di = (T*)malloc(sizeof(T)*hDim*batchSize);
	df = (T*)malloc(sizeof(T)*hDim*batchSize);
	dgifo_lin = (T*)malloc(sizeof(T)*gifoDim*batchSize);
	// Information down to time period
	dhm = (T*)malloc(sizeof(T)*hDim*batchSize);
	dsm = (T*)malloc(sizeof(T)*hDim*batchSize);
	// Information down to the lower layer
	dx_t = (T*)malloc(sizeof(T)*xDim*batchSize*periods);

	// Temporary and constant space
	g_temp = (T*)malloc(sizeof(T)*hDim*batchSize);

	ONES_BATCH_SIZE = (T*)malloc(sizeof(T)*batchSize);
	ZEROS_H_STRIDE = (T*)malloc(sizeof(T)*hDim*batchSize);
}

template<>
int LstmLayer<float>::forward_pass(int t)
{
	if (!weights_x | !weights_h | !biases)
	{
		return RNN_ERR_WEIGHTS_UNINITIATED;
	}

	int xStride = batchSize*xDim;
	int hStride = batchSize*hDim;
	int gifoStride = batchSize*gifoDim;
	// Convert to 0 based
	int tidx = t - 1;

	// Extract time t variable
	float* x = x_t + tidx*xStride;
	float* gifo_lin = gifo_lin_t + tidx*gifoStride;
	float* g = g_t + tidx*hStride;
	float* ii = i_t + tidx*hStride;
	float* f = f_t + tidx*hStride;
	float* o = o_t + tidx*hStride;
	float* h = h_t + tidx*hStride;
	float* s = s_t + tidx*hStride;
	float* tanhs = tanhs_t + tidx*hStride;

	// Forward pass one time
	float* hm;
	float* sm;
	if (t > 1)
	{
		hm = h_t + (tidx - 1)*hStride;
		sm = s_t + (tidx - 1)*hStride;
	}
	else
	{
		hm = ZEROS_H_STRIDE;
		sm = ZEROS_H_STRIDE;
	}

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, gifoDim, batchSize, xDim, 1, weights_x, gifoDim, x, xDim, 0, gifo_lin, gifoDim);
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, gifoDim, batchSize, hDim, 1, weights_h, gifoDim, hm, hDim, 1, gifo_lin, gifoDim);
	// Plus biases
	for (int j = 0; j < batchSize; j++)
	{
		float* gifo_lin_j = gifo_lin + j*gifoDim;
#pragma simd
		for (int k = 0; k < gifoDim; k++)
			gifo_lin_j[k] += biases[k];

		memcpy(g_lin + j*hDim, gifo_lin + j*gifoDim + 0 * hDim, sizeof(float)*hDim);
		memcpy(i_lin + j*hDim, gifo_lin + j*gifoDim + 1 * hDim, sizeof(float)*hDim);
		memcpy(f_lin + j*hDim, gifo_lin + j*gifoDim + 2 * hDim, sizeof(float)*hDim);
		memcpy(o_lin + j*hDim, gifo_lin + j*gifoDim + 3 * hDim, sizeof(float)*hDim);
	}

#pragma simd
	for (int k = 0; k < hStride; k++)
	{
		g[k] = tanh(g_lin[k]);
		ii[k] = SIGMOID(i_lin[k]);
		f[k] = SIGMOID(f_lin[k]);
		o[k] = SIGMOID(o_lin[k]);
		s[k] = g[k] * ii[k] + sm[k] * f[k];
		tanhs[k] = tanh(s[k]);
		h[k] = tanhs[k] * o[k];
	}

	return 0;
}

template<>
int LstmLayer<float>::back_propagation(int t)
{
	if (batchSize > batchSize)
	{
		return RNN_ERR_ALLOC_BATCH_SIZE_TOO_SMALL;
	}

	if (!dweights_x | !dweights_h | !dbiases)
	{
		return RNN_ERR_DWEIGHTS_UNINITIATED;
	}

	int xStride = batchSize*xDim;
	int hStride = batchSize*hDim;
	int gifoStride = batchSize*gifoDim;
	// Convert to 0 based
	int tidx = t - 1;

	float* hm;
	float* sm;
	if (t == 1)
	{
		hm = ZEROS_H_STRIDE;
		sm = ZEROS_H_STRIDE;
	}
	else
	{
		hm = h_t + (tidx - 1)*hStride;
		sm = s_t + (tidx - 1)*hStride;
	}

	// Extract time t variable
	float* x = x_t + tidx*xStride;
	float* g = g_t + tidx*hStride;
	float* ii = i_t + tidx*hStride;
	float* f = f_t + tidx*hStride;
	float* o = o_t + tidx*hStride;
	float* h = h_t + tidx*hStride;
	float* s = s_t + tidx*hStride;
	float* tanhs = tanhs_t + tidx*hStride;

	float* dtop = dtop_t + tidx*hStride;
	float* dx = dx_t + tidx*xStride;

	// Back propagation for one time step
	// Transpose all relevant matrix
	mkl_somatcopy('C', 'T', hDim, batchSize, 1, g, hDim, g_temp, batchSize);
	memcpy(g, g_temp, sizeof(float)*hStride);

	mkl_somatcopy('C', 'T', hDim, batchSize, 1, ii, hDim, g_temp, batchSize);
	memcpy(ii, g_temp, sizeof(float)*hStride);

	mkl_somatcopy('C', 'T', hDim, batchSize, 1, f, hDim, g_temp, batchSize);
	memcpy(f, g_temp, sizeof(float)*hStride);

	mkl_somatcopy('C', 'T', hDim, batchSize, 1, o, hDim, g_temp, batchSize);
	memcpy(o, g_temp, sizeof(float)*hStride);

	mkl_somatcopy('C', 'T', hDim, batchSize, 1, sm, hDim, g_temp, batchSize);
	memcpy(sm, g_temp, sizeof(float)*hStride);

	mkl_somatcopy('C', 'T', hDim, batchSize, 1, tanhs, hDim, g_temp, batchSize);
	memcpy(tanhs, g_temp, sizeof(float)*hStride);

	// dh = dh + dupper
	if (t == periods) {
		// Last period
#pragma simd
		for (int j = 0; j < hStride; j++)
		{
			dh[j] = dtop[j];
			ds[j] = 0;
		}
	}
	else
	{
#pragma simd
		for (int j = 0; j < hStride; j++)
		{
			dh[j] = dhm[j] + dtop[j];
			ds[j] = dsm[j];
		}
	}

	/*
	do = dh.*tanhs';
	dtanhs = dh.*o';
	ds = ds + dtanhs.*(1-tanhs.^2)';
	dg = ds.*ii';
	di = ds.*g';
	df = ds.*sm';
	dsm = ds.*f';
	dglin = dg.*(1-g.^2)';
	dilin = di.*(ii.*(1-ii))';
	dflin = df.*(f.*(1-f))';
	dolin = do.*(o.*(1-o))';
	dgifo_lin = [dglin,dilin,dflin,dolin];
	*/
	float* dg_lin = dgifo_lin + 0 * hStride;
	float* di_lin = dgifo_lin + 1 * hStride;
	float* df_lin = dgifo_lin + 2 * hStride;
	float* doo_lin = dgifo_lin + 3 * hStride;

#pragma simd
	for (int j = 0; j < hStride; j++)
	{
		doo[j] = dh[j] * tanhs[j];
		dtanhs[j] = dh[j] * o[j];
		ds[j] += dtanhs[j] * (1 - tanhs[j] * tanhs[j]);
		dg[j] = ds[j] * ii[j];
		di[j] = ds[j] * g[j];
		dsm[j] = ds[j] * f[j];
		df[j] = ds[j] * sm[j];

		dg_lin[j] = dg[j] * (1 - g[j] * g[j]);
		di_lin[j] = di[j] * (ii[j] * (1 - ii[j]));
		df_lin[j] = df[j] * (f[j] * (1 - f[j]));
		doo_lin[j] = doo[j] * (o[j] * (1 - o[j]));
	}

	// dW_x = dW_x + dgifo_lin'*x';
	cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans, gifoDim, xDim, batchSize, 1, dgifo_lin, batchSize, x, xDim, 1, dweights_x, gifoDim);
	// dWh = dW_h + dgifo_lin'*hm';
	cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans, gifoDim, hDim, batchSize, 1, dgifo_lin, batchSize, hm, hDim, 1, dweights_h, gifoDim);
	// dbias = dbas + sum(dgifo_lin,1)';
	cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, gifoDim, 1, batchSize, 1, dgifo_lin, batchSize, ONES_BATCH_SIZE, batchSize, 1, dbiases, gifoDim);

	// dhm = dgifo_lin * W_h;
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, batchSize, hDim, gifoDim, 1, dgifo_lin, batchSize, weights_h, gifoDim, 0, dhm, batchSize);
	// dx = dgifo_lin * W_x
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, batchSize, xDim, gifoDim, 1, dgifo_lin, batchSize, weights_x, gifoDim, 0, dx, batchSize);

	return 0;
}

#pragma endregion LstmLayer

#pragma region SoftmaxLayer

template <typename T>
class SoftmaxLayer {
private:
	T* memory;

public:
	int hDim;
	int yDim;
	int periods;
	int batchSize;

	int hasPreallocSpace;
	int memorySize;

	int isInit;

	// Constructor
	SoftmaxLayer() : hDim(0), yDim(0), periods(0), batchSize(0), hasPreallocSpace(0), memorySize(0), isInit(0){};
	SoftmaxLayer(int _hDim, int _yDim, int _periods, int _batchSize, T* _prealloc_space = 0)
		: hDim(_hDim), yDim(_yDim), periods(_periods), batchSize(_batchSize), memory(_prealloc_space), isInit(0)
	{
		hasPreallocSpace = (_prealloc_space != 0);
		init();
	}

	void init()
	{
		memorySize = computeMemorySize();

		// Allocate space
		if (memory != 0)
		{
			assign_memory();
		}
		else
		{
			alloc();
		}

		// Assign constant
		for (int j = 0; j < batchSize; j++)
		{
			ONES_BATCH_SIZE[j] = 1;
		}

		isInit = 1;
	}

	~SoftmaxLayer()
	{
		dealloc();
	}

	// weights
	T* weights;
	T* biases;

	// Input
	T* h_t;

	// Output
	T* y_t;

	// Used in forward pass
	T* ylin_t;
	T* yhat_t;
	T* loss_t;

	// Used in back propagation
	T* dyhat;
	T* dh_t;

	// Derivative
	T* dweights;
	T* dbiases;

	// Constant used in computation
	T* ONES_BATCH_SIZE;

	void dealloc();

	void alloc();
	void assign_memory();
	int computeMemorySize();

	void init_dweights()
	{
		memset(dweights, 0, sizeof(T)*hDim*yDim);
		memset(dbiases, 0, sizeof(T)*yDim);
	}

	int forward_pass(int t, double temperature = 1);

	int back_propagation(int t);
};

template<typename T>
void SoftmaxLayer<T>::dealloc()
{
	if (hasPreallocSpace)
		return;

	free(h_t);
	free(y_t);
	free(ylin_t);
	free(yhat_t);
	free(loss_t);

	// Back
	free(dyhat);
	free(dh_t);

	// Constant
	free(ONES_BATCH_SIZE);
}

template<typename T>
int SoftmaxLayer<T>::computeMemorySize()
{
	int ptr = 0;

	// h_t = ptr;
	ptr += hDim*batchSize*periods;
	// y_t = ptr;
	ptr += yDim*batchSize*periods;
	// ylin_t = ptr;
	ptr += yDim*batchSize*periods;
	// yhat_t = ptr;
	ptr += yDim*batchSize*periods;
	// loss_t = ptr;
	ptr += batchSize*periods;

	// Back
	// dyhat = ptr;
	ptr += yDim*batchSize;
	// dh_t = ptr;
	ptr += hDim*batchSize*periods;

	// Constant
	// ONES_BATCH_SIZE = ptr;
	ptr += batchSize;

	return ptr;
}

template<typename T>
void SoftmaxLayer<T>::assign_memory()
{
	// Forward
	T* ptr = memory;

	h_t = ptr;
	ptr += hDim*batchSize*periods;
	y_t = ptr;
	ptr += yDim*batchSize*periods;
	ylin_t = ptr;
	ptr += yDim*batchSize*periods;
	yhat_t = ptr;
	ptr += yDim*batchSize*periods;
	loss_t = ptr;
	ptr += batchSize*periods;

	// Back
	dyhat = ptr;
	ptr += yDim*batchSize;
	dh_t = ptr;
	ptr += hDim*batchSize*periods;

	// Constant
	ONES_BATCH_SIZE = ptr;
	ptr += batchSize;
}

template<typename T>
void SoftmaxLayer<T>::alloc()
{
	if (memory != 0)
		return;

	h_t = (T*)malloc(sizeof(T)*hDim*batchSize*periods);
	y_t = (T*)malloc(sizeof(T)*yDim*batchSize*periods);
	ylin_t = (T*)malloc(sizeof(T)*yDim*batchSize*periods);
	yhat_t = (T*)malloc(sizeof(T)*yDim*batchSize*periods);
	loss_t = (T*)malloc(sizeof(T)*batchSize*periods);

	// Back
	dyhat = (T*)malloc(sizeof(T)*yDim*batchSize);
	dh_t = (T*)malloc(sizeof(T)*hDim*batchSize*periods);

	// Constant
	ONES_BATCH_SIZE = (T*)malloc(sizeof(T)*batchSize);
}

template<>
int SoftmaxLayer<float>::forward_pass(int t, double temperature)
{
	if (!weights | !biases)
	{
		return RNN_ERR_WEIGHTS_UNINITIATED;
	}

	int yStride = batchSize*yDim;
	int hStride = batchSize*hDim;
	// Convert to 0 based
	int tidx = t - 1;

	float* h = h_t + tidx*hStride;
	float* y = y_t + tidx*yStride;
	float* ylin = ylin_t + tidx*yStride;
	float* yhat = yhat_t + tidx*yStride;
	float* loss = loss_t + tidx*batchSize;

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, yDim, batchSize, hDim, 1, weights, yDim, h, hDim, 0, ylin, yDim);

	for (int j = 0; j < batchSize; j++)
	{
		/*
		float* ylin_j = ylin + j*yDim;
		float* yhat_j = yhat + j*yDim;
		float* y_j = y + j*yDim;
		float* loss_j = loss + j;
		*/
#pragma novector
		for (int k = 0; k < yDim; k++)
		{
			ylin[j*yDim + k] += biases[k];
		}

		float ylin_max = -1e20;
#pragma novector
		for (int k = 0; k < yDim; k++)
		{
			if (ylin[j*yDim + k] > ylin_max)
				ylin_max = ylin[j*yDim + k];
		}

		float sum_exp_ylin_minus_max = 0;

#pragma novector
		for (int k = 0; k < yDim; k++)
		{
			yhat[j*yDim + k] = exp((ylin[j*yDim + k] - ylin_max) / temperature);
			sum_exp_ylin_minus_max += yhat[j*yDim + k];
		}

		float loss_sum = 0;
#pragma novector
		for (int k = 0; k < yDim; k++)
		{
			yhat[j*yDim + k] /= sum_exp_ylin_minus_max;
			loss_sum += -y[j*yDim + k] * log(yhat[j*yDim + k]);
		}
		loss[j] = loss_sum;
	}

	return 0;
}

template<>
int SoftmaxLayer<float>::back_propagation(int t)
{
	if (!dweights | !dbiases)
	{
		return RNN_ERR_DWEIGHTS_UNINITIATED;
	}

	int yStride = batchSize*yDim;
	int hStride = batchSize*hDim;
	// Convert to 0 based
	int tidx = t - 1;

	float* h = h_t + tidx*hStride;
	float* y = y_t + tidx*yStride;
	float* ylin = ylin_t + tidx*yStride;
	float* yhat = yhat_t + tidx*yStride;
	float* dh = dh_t + tidx*hStride;

	for (int k = 0; k < yDim*batchSize; k++)
	{
		ylin[k] = yhat[k] - y[k];
	}
	// Transpose matrix
	mkl_somatcopy('C', 'T', yDim, batchSize, 1, ylin, yDim, dyhat, batchSize);

	// dWyh = dWyh + dyhat'*h';
	cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans, yDim, hDim, batchSize, 1, dyhat, batchSize, h, hDim, 1, dweights, yDim);

	// dby = dby + sum(dyhat,1)';
	cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, yDim, 1, batchSize, 1, dyhat, batchSize, ONES_BATCH_SIZE, batchSize, 1, dbiases, yDim);

	// dh = dyhat * Wyh;
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, batchSize, hDim, yDim, 1, dyhat, batchSize, weights, yDim, 0, dh, batchSize);

	return 0;
}

#pragma endregion SoftmaxLayer
