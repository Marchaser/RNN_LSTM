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

#if defined(TYPENAME_DOUBLE)
#define gemm cblas_dgemm
#define omatcopy mkl_domatcopy
#elif defined(TYPENAME_FLOAT)
#define gemm cblas_sgemm
#define omatcopy mkl_somatcopy
#endif


class NNLayer {
public:
	int periods;
	int batchSize;
	virtual int forward_pass(int t) = 0;
	virtual int back_propagation(int t) = 0;

	virtual int forward_pass_T()
	{
		int status;
		for (int t = 1; t <= periods; t++)
		{
			status = forward_pass(t);
		}
		return status;
	}

	virtual int back_propagation_T()
	{
		int status;
		for (int t = periods; t >= 1; t--)
		{
			status = back_propagation(t);
		}
		return status;
	}
};

#pragma region LstmLayer

template <typename T>
class LstmLayer : public NNLayer {
public:
	int inputDim;
	int outputDim;
	int gifoDim;

	int batchSize;
	int sizeWeights;
	int outputStride;

	// Static
	static int compute_size_weights(int inputDim, int outputDim) { return outputDim * 4 * (inputDim + outputDim + 1); }

	// Clear information
	void clear_info()
	{
		memset(hm_0, 0, sizeof(T)*outputDim*batchSize);
		memset(sm_0, 0, sizeof(T)*outputDim*batchSize);
	}

	// Constructor
	LstmLayer(int _inputDim, int _outputDim, int _periods, int _batchSize)
		: inputDim(_inputDim), outputDim(_outputDim)
	{
		periods = _periods;
		batchSize = _batchSize;

		gifoDim = outputDim * 4;

		outputStride = outputDim*batchSize;

		sizeWeights = compute_size_weights(inputDim, outputDim);

		alloc();
		// Set constant
		for (int j = 0; j < batchSize; j++)
		{
			ONES_BATCH_SIZE[j] = 1;
		}

		// Set initial information at construction
		clear_info();
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
	T* x_t_reserve;
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
	// Information kept from last train
	T* hm_0;
	T* sm_0;

	// Used for backward propagation
	// Information from the upper layer
	T* dtop_t_reserve;
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
	T* temp;
	// Constants used for computation
	T* ONES_BATCH_SIZE;

	void init_dweights()
	{
		memset(dweights_x, 0, sizeof(T)*gifoDim*inputDim);
		memset(dweights_h, 0, sizeof(T)*gifoDim*outputDim);
		memset(dbiases, 0, sizeof(T)*gifoDim);
	}


	T* get_f_biases()
	{
		return biases + 2 * outputDim;
	}

	void assign_weights(T** p);
	void assign_dweights(T** p);

	int forward_pass(int t);
	int back_propagation(int t);

	void dealloc();
	void alloc();

	/// Store information for next training
	void store_info(T* _hm_0, T* _sm_0);

	/// Fetch information
	void fetch_from_bottom_ptr(T* src);
	void fetch_from_bottom(T* src);
	void fetch_from_bottom_with_dropout(T* src, T dropoutRate);
	void fetch_from_top_ptr(T* src);
	void fetch_from_top(T* src);
};

template<typename T>
void LstmLayer<T>::fetch_from_bottom_with_dropout(T* src, T dropoutRate)
{
	x_t = x_t_reserve;
	memcpy(x_t, src, sizeof(T)*inputDim*batchSize*periods);
	T keepRate = 1 - dropoutRate;
#pragma simd
	for (int j = 0; j < inputDim*batchSize*periods; j++)
	{
		x_t[j] *= keepRate;
	}
}

template<typename T>
void LstmLayer<T>::fetch_from_bottom_ptr(T* src)
{
	x_t = src;
}

template<typename T>
void LstmLayer<T>::fetch_from_bottom(T* src)
{
	x_t = x_t_reserve;
	memcpy(x_t, src, sizeof(T)*inputDim*batchSize*periods);
}

template<typename T>
void LstmLayer<T>::fetch_from_top_ptr(T* src)
{
	dtop_t = src;
}

template<typename T>
void LstmLayer<T>::fetch_from_top(T* src)
{
	dtop_t = dtop_t_reserve;
	memcpy(dtop_t, src, sizeof(T)*outputDim*batchSize*periods);
}

template<typename T>
void LstmLayer<T>::store_info(T* _hm_0, T* _sm_0)
{
	// Copy h(1) to hm_0
	memcpy(hm_0, _hm_0, sizeof(T)*outputDim*batchSize);

	// Copy s(1) to sm_0
	// This may look fishy, but remember transpose is done for sm in g_temp
	memcpy(sm_0, _sm_0, sizeof(T)*outputDim*batchSize);
}



template<typename T>
void LstmLayer<T>::assign_weights(T** p)
{
	weights_x = *p;
	*p += gifoDim*inputDim;
	weights_h = *p;
	*p += gifoDim*outputDim;
	biases = *p;
	*p += gifoDim;
}

template<typename T>
void LstmLayer<T>::assign_dweights(T** p)
{
	dweights_x = *p;
	*p += gifoDim*inputDim;
	dweights_h = *p;
	*p += gifoDim*outputDim;
	dbiases = *p;
	*p += gifoDim;
}

template<typename T>
void LstmLayer<T>::dealloc()
{
	// Allocate memory
	delete[] x_t_reserve;

	delete[] gifo_lin_t;

	delete[] g_lin;
	delete[] i_lin;
	delete[] f_lin;
	delete[] o_lin;

	delete[] g_t;
	delete[] i_t;
	delete[] f_t;
	delete[] o_t;
	delete[] h_t;
	delete[] s_t;
	delete[] hm_0;
	delete[] sm_0;
	delete[] tanhs_t;

	// Used for backward propagation
	// Information from the upper layer
	delete[] dtop_t_reserve;
	// Information within current period
	delete[] dh;
	delete[] ds;
	delete[] doo;
	delete[] dtanhs;
	delete[] dg;
	delete[] di;
	delete[] df;
	delete[] dgifo_lin;
	// Information down to time period
	delete[] dhm;
	delete[] dsm;
	// Information down to the lower layer
	delete[] dx_t;

	// Temporary and constant space
	delete[] temp;

	delete[] ONES_BATCH_SIZE;
}

template<typename T>
void LstmLayer<T>::alloc()
{
	// Allocate memory
	x_t_reserve = new T[inputDim*batchSize*periods];
	x_t = x_t_reserve;

	gifo_lin_t = new T[gifoDim*batchSize*periods];

	g_lin = new T[outputDim*batchSize];
	i_lin = new T[outputDim*batchSize];
	f_lin = new T[outputDim*batchSize];
	o_lin = new T[outputDim*batchSize];

	g_t = new T[outputDim*batchSize*periods];
	i_t = new T[outputDim*batchSize*periods];
	f_t = new T[outputDim*batchSize*periods];
	o_t = new T[outputDim*batchSize*periods];
	h_t = new T[outputDim*batchSize*periods];
	s_t = new T[outputDim*batchSize*periods];
	hm_0 = new T[outputDim*batchSize];
	sm_0 = new T[outputDim*batchSize];
	tanhs_t = new T[outputDim*batchSize*periods];

	// Used for backward propagation
	// Information from the upper layer
	dtop_t_reserve = new T[outputDim*batchSize*periods];
	dtop_t = dtop_t_reserve;
	// Information within current period
	dh = new T[outputDim*batchSize];
	ds = new T[outputDim*batchSize];
	doo = new T[outputDim*batchSize];
	dtanhs = new T[outputDim*batchSize];
	dg = new T[outputDim*batchSize];
	di = new T[outputDim*batchSize];
	df = new T[outputDim*batchSize];
	dgifo_lin = new T[gifoDim*batchSize];
	// Information down to time period
	dhm = new T[outputDim*batchSize];
	dsm = new T[outputDim*batchSize];
	// Information down to the lower layer
	dx_t = new T[inputDim*batchSize*periods];

	// Temporary and constant space
	temp = new T[outputDim*batchSize];

	ONES_BATCH_SIZE = new T[batchSize];
}

template<typename T>
int LstmLayer<T>::forward_pass(int t)
{
	if (!weights_x | !weights_h | !biases)
	{
		return RNN_ERR_WEIGHTS_UNINITIATED;
	}

	int inputStride = batchSize*inputDim;
	int outputStride = batchSize*outputDim;
	int gifoStride = batchSize*gifoDim;
	// Convert to 0 based
	int tidx = t - 1;

	// Extract time t variable
	T* x = x_t + tidx*inputStride;
	T* gifo_lin = gifo_lin_t + tidx*gifoStride;
	T* g = g_t + tidx*outputStride;
	T* ii = i_t + tidx*outputStride;
	T* f = f_t + tidx*outputStride;
	T* o = o_t + tidx*outputStride;
	T* h = h_t + tidx*outputStride;
	T* s = s_t + tidx*outputStride;
	T* tanhs = tanhs_t + tidx*outputStride;

	// Forward pass one time
	T* hm;
	T* sm;
	if (t > 1)
	{
		hm = h_t + (tidx - 1)*outputStride;
		sm = s_t + (tidx - 1)*outputStride;
	}
	else
	{
		hm = hm_0;
		sm = sm_0;
	}

	gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, gifoDim, batchSize, inputDim, 1, weights_x, gifoDim, x, inputDim, 0, gifo_lin, gifoDim);
	gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, gifoDim, batchSize, outputDim, 1, weights_h, gifoDim, hm, outputDim, 1, gifo_lin, gifoDim);
	// Plus biases
	for (int j = 0; j < batchSize; j++)
	{
		T* gifo_lin_j = gifo_lin + j*gifoDim;
#pragma simd
		for (int k = 0; k < gifoDim; k++)
			gifo_lin_j[k] += biases[k];

		memcpy(g_lin + j*outputDim, gifo_lin + j*gifoDim + 0 * outputDim, sizeof(T)*outputDim);
		memcpy(i_lin + j*outputDim, gifo_lin + j*gifoDim + 1 * outputDim, sizeof(T)*outputDim);
		memcpy(f_lin + j*outputDim, gifo_lin + j*gifoDim + 2 * outputDim, sizeof(T)*outputDim);
		memcpy(o_lin + j*outputDim, gifo_lin + j*gifoDim + 3 * outputDim, sizeof(T)*outputDim);
	}

#pragma simd
	for (int k = 0; k < outputStride; k++)
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

template<typename T>
int LstmLayer<T>::back_propagation(int t)
{
	if (!dweights_x | !dweights_h | !dbiases)
	{
		return RNN_ERR_DWEIGHTS_UNINITIATED;
	}

	int inputStride = batchSize*inputDim;
	int hStride = batchSize*outputDim;
	int gifoStride = batchSize*gifoDim;
	// Convert to 0 based
	int tidx = t - 1;

	T* hm;
	T* sm;
	if (t == 1)
	{
		hm = hm_0;
		sm = sm_0;
	}
	else
	{
		hm = h_t + (tidx - 1)*hStride;
		sm = s_t + (tidx - 1)*hStride;
	}

	// Extract time t variable
	T* x = x_t + tidx*inputStride;
	T* g = g_t + tidx*hStride;
	T* ii = i_t + tidx*hStride;
	T* f = f_t + tidx*hStride;
	T* o = o_t + tidx*hStride;
	T* h = h_t + tidx*hStride;
	T* s = s_t + tidx*hStride;
	T* tanhs = tanhs_t + tidx*hStride;

	T* dtop = dtop_t + tidx*hStride;
	T* dx = dx_t + tidx*inputStride;

	// Back propagation for one time step
	// Transpose all relevant matrix
	omatcopy('C', 'T', outputDim, batchSize, 1, g, outputDim, temp, batchSize);
	memcpy(g, temp, sizeof(T)*hStride);

	omatcopy('C', 'T', outputDim, batchSize, 1, ii, outputDim, temp, batchSize);
	memcpy(ii, temp, sizeof(T)*hStride);

	omatcopy('C', 'T', outputDim, batchSize, 1, f, outputDim, temp, batchSize);
	memcpy(f, temp, sizeof(T)*hStride);

	omatcopy('C', 'T', outputDim, batchSize, 1, o, outputDim, temp, batchSize);
	memcpy(o, temp, sizeof(T)*hStride);

	omatcopy('C', 'T', outputDim, batchSize, 1, tanhs, outputDim, temp, batchSize);
	memcpy(tanhs, temp, sizeof(T)*hStride);

	// g_temp is finally used to store sm
	omatcopy('C', 'T', outputDim, batchSize, 1, sm, outputDim, temp, batchSize);
	sm = temp;

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
	T* dg_lin = dgifo_lin + 0 * hStride;
	T* di_lin = dgifo_lin + 1 * hStride;
	T* df_lin = dgifo_lin + 2 * hStride;
	T* doo_lin = dgifo_lin + 3 * hStride;

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
	gemm(CblasColMajor, CblasTrans, CblasTrans, gifoDim, inputDim, batchSize, 1, dgifo_lin, batchSize, x, inputDim, 1, dweights_x, gifoDim);
	// dWh = dW_h + dgifo_lin'*hm';
	gemm(CblasColMajor, CblasTrans, CblasTrans, gifoDim, outputDim, batchSize, 1, dgifo_lin, batchSize, hm, outputDim, 1, dweights_h, gifoDim);
	// dbias = dbas + sum(dgifo_lin,1)';
	gemm(CblasColMajor, CblasTrans, CblasNoTrans, gifoDim, 1, batchSize, 1, dgifo_lin, batchSize, ONES_BATCH_SIZE, batchSize, 1, dbiases, gifoDim);

	// dhm = dgifo_lin * W_h;
	gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, batchSize, outputDim, gifoDim, 1, dgifo_lin, batchSize, weights_h, gifoDim, 0, dhm, batchSize);
	// dx = dgifo_lin * W_x
	gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, batchSize, inputDim, gifoDim, 1, dgifo_lin, batchSize, weights_x, gifoDim, 0, dx, batchSize);

	return 0;
}

#pragma endregion LstmLayer

#pragma region LinearLayer

template <typename T>
class LinearLayer : public NNLayer {
public:
	int inputDim;
	int outputDim;
	int batchSize;

	int sizeWeights;

	// Static
	static int compute_size_weights(int hDim, int yDim) {
		return yDim*(hDim + 1);
	}

	// Constructor
	LinearLayer(int _inputDim, int _outputDim, int _periods, int _batchSize)
		: inputDim(_inputDim), outputDim(_outputDim)
	{
		periods = _periods;
		batchSize = _batchSize;

		sizeWeights = compute_size_weights(inputDim, outputDim);

		alloc();
		// Assign constant
		for (int j = 0; j < batchSize; j++)
		{
			ONES_BATCH_SIZE[j] = 1;
		}
	}


	~LinearLayer()
	{
		dealloc();
	}

	// weights
	T* weights;
	T* biases;

	// Input
	T* x_t_reserve;
	T* x_t;

	// Output
	T* y_t;

	// Used in back propagation
	T* dy_t_reserve;
	T* dy_t;
	T* dx_t;

	// Derivative
	T* dweights;
	T* dbiases;

	// Constant used in computation
	T* ONES_BATCH_SIZE;

	void dealloc();
	void alloc();

	void assign_weights(T** p);
	void assign_dweights(T** p);

	void init_dweights()
	{
		memset(dweights, 0, sizeof(T)*inputDim*outputDim);
		memset(dbiases, 0, sizeof(T)*outputDim);
	}

	int forward_pass(int t);

	int back_propagation(int t);

	void fetch_from_bottom(T* src);
	void fetch_from_bottom_ptr(T* src);
	void fetch_from_bottom_with_dropout(T* src, T dropoutRate);
	// Top is the output, since it's a final layer
	void fetch_from_top(T* src);
	void fetch_from_top_ptr(T* src);
};

template<typename T>
void LinearLayer<T>::fetch_from_bottom_with_dropout(T* src, T dropoutRate)
{
	x_t = x_t_reserve;
	memcpy(x_t, src, sizeof(T)*inputDim*batchSize*periods);
	T keepRate = 1 - dropoutRate;
#pragma simd
	for (int j = 0; j < inputDim*batchSize*periods; j++)
	{
		x_t[j] *= keepRate;
	}
}

template<typename T>
void LinearLayer<T>::fetch_from_bottom_ptr(T* src)
{
	x_t = src;
}

template<typename T>
void LinearLayer<T>::fetch_from_bottom(T* src)
{
	x_t = x_t_reserve;
	memcpy(x_t, src, sizeof(T)*inputDim*batchSize*periods);
}

template<typename T>
void LinearLayer<T>::fetch_from_top_ptr(T* src)
{
	dy_t = src;
}

template<typename T>
void LinearLayer<T>::fetch_from_top(T* src)
{
	memcpy(dy_t, src, sizeof(T)*outputDim*batchSize*periods);
}


template<typename T>
void LinearLayer<T>::assign_weights(T** p)
{
	weights = *p;
	*p += outputDim*inputDim;
	biases = *p;
	*p += outputDim;
}

template<typename T>
void LinearLayer<T>::assign_dweights(T** p)
{
	dweights = *p;
	*p += outputDim*inputDim;
	dbiases = *p;
	*p += outputDim;
}

template<typename T>
void LinearLayer<T>::dealloc()
{
	delete[] x_t_reserve;
	delete[] y_t;

	// Back
	delete[] dy_t_reserve;
	delete[] dx_t;

	// Constant
	delete[] ONES_BATCH_SIZE;
}

template<typename T>
void LinearLayer<T>::alloc()
{
	x_t_reserve = new T[inputDim*batchSize*periods];
	x_t = x_t_reserve;
	y_t = new T[outputDim*batchSize*periods];

	// Back
	dy_t_reserve = new T[outputDim*batchSize*periods];
	dy_t = dy_t_reserve;
	dx_t = new T[inputDim*batchSize*periods];

	// Constant
	ONES_BATCH_SIZE = new T[batchSize];
}

template<typename T>
int LinearLayer<T>::forward_pass(int t)
{
	if (!weights | !biases)
	{
		return RNN_ERR_WEIGHTS_UNINITIATED;
	}

	int yStride = batchSize*outputDim;
	int xStride = batchSize*inputDim;
	// Convert to 0 based
	int tidx = t - 1;

	T* x = x_t + tidx*xStride;
	T* y = y_t + tidx*yStride;

	gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, outputDim, batchSize, inputDim, 1, weights, outputDim, x, inputDim, 0, y, outputDim);

	for (int j = 0; j < batchSize; j++)
	{
		T* y_j = y + j*outputDim;
#pragma simd
		for (int k = 0; k < outputDim; k++)
		{
			y_j[k] += biases[k];
		}
	}

	return 0;
}

template<typename T>
int LinearLayer<T>::back_propagation(int t)
{
	if (!dweights | !dbiases)
	{
		return RNN_ERR_DWEIGHTS_UNINITIATED;
	}

	int xStride = batchSize*inputDim;
	int yStride = batchSize*outputDim;
	// Convert to 0 based
	int tidx = t - 1;

	T* x = x_t + tidx*xStride;
	T* y = y_t + tidx*yStride;
	T* dy = dy_t + tidx*yStride;
	T* dx = dx_t + tidx*xStride;

	// dWyx = dWyh + dy'*x';
	gemm(CblasColMajor, CblasTrans, CblasTrans, outputDim, inputDim, batchSize, 1, dy, batchSize, x, inputDim, 1, dweights, outputDim);

	// dby = dby + sum(dy,1)';
	gemm(CblasColMajor, CblasTrans, CblasNoTrans, outputDim, 1, batchSize, 1, dy, batchSize, ONES_BATCH_SIZE, batchSize, 1, dbiases, outputDim);

	// dx = dy * Wyh;
	gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, batchSize, inputDim, outputDim, 1, dy, batchSize, weights, outputDim, 0, dx, batchSize);

	return 0;
}

#pragma endregion LinearLayer

#pragma region DropoutLayer
template <typename T>
class DropoutLayer : public NNLayer {
public:
	int inputDim;
	int batchSize;
	int* D_t;
	T* x_t;
	T* dx_t;
	double dropoutRate;
	int seed;

	DropoutLayer(int _hDim, int _periods, int _batchSize, int _seed = 823, double _dropoutRate = 0.5)
		: inputDim(_hDim), seed(_seed), dropoutRate(_dropoutRate)
	{
		periods = _periods;
		batchSize = _batchSize;
		D_t = (int*)malloc(sizeof(int)*inputDim*batchSize*periods);
		x_t = (T*)malloc(sizeof(T)*inputDim*batchSize*periods);
		dx_t = (T*)malloc(sizeof(T)*inputDim*batchSize*periods);
	}

	~DropoutLayer()
	{
		free(D_t);
		free(x_t);
		free(dx_t);
	}

	// Forward
	int forward_pass(int t);
	// override
	int forward_pass_T();

	// Back
	int back_propagation(int t);
	// override
	int back_propagation_T();

	// Utility
	inline int fastrand()
	{
		seed = (214013 * seed + 2531011);
		return (seed >> 16) & 0x7FFF;
	}

	void fetch_from_bottom(T* src);
	void fetch_from_top(T* src);
};

template <typename T>
void DropoutLayer<T>::fetch_from_bottom(T* src)
{
	memcpy(x_t, src, sizeof(T)*inputDim*batchSize*periods);
}

template <typename T>
void DropoutLayer<T>::fetch_from_top(T* src)
{
	memcpy(dx_t, src, sizeof(T)*inputDim*batchSize*periods);
}

template <typename T>
int DropoutLayer<T>::forward_pass_T()
{
	int inputStride = inputDim*batchSize;

	int* D = D_t;
	for (int j = 0; j < inputStride; j++)
	{
		double dropoutDraws = (double)fastrand() / (double)(0x7FFF);
		if (dropoutDraws > dropoutRate)
		{
			// Keep observations
			D[j] = 1;
		}
		else
		{
			// Drop out
			D[j] = 0;
		}
	}

	for (int t = 1; t <= periods; t++)
	{
		// Extract time t variable
		int tidx = t - 1;
		T* x = x_t + inputStride*tidx;

		for (int j = 0; j < inputStride; j++)
		{
			if (D[j] == 0)
				x[j] = 0;
		}
	}

	return 0;
}

template <typename T>
int DropoutLayer<T>::forward_pass(int t)
{
	int inputStride = inputDim*batchSize;

	// Extract time t variable
	int tidx = t - 1;
	int* D = D_t + inputStride*tidx;
	T* x = x_t + inputStride*tidx;

	for (int j = 0; j < inputStride; j++)
	{
		double dropoutDraws = (double)fastrand() / (double)RAND_MAX;
		if (dropoutDraws > dropoutRate)
		{
			// Keep observations
			D[j] = 1;
		}
		else
		{
			// Drop out
			D[j] = 0;
			x[j] = 0;
		}
	}

	return 0;
}

template <typename T>
int DropoutLayer<T>::back_propagation_T()
{
	int inputStride = inputDim*batchSize;

	// Extract time t variable
	int* D = D_t;

	for (int t = periods; t >= 1; t--)
	{
		int tidx = t - 1;
		T* dx = dx_t + inputStride*tidx;

		for (int j = 0; j < inputDim; j++)
		{
			for (int k = 0; k < batchSize; k++)
			{
				// Recall size of D is [hDim, batchSize]
				// Size of dh is [batchSize, hDim]
				if (D[k*inputDim + j] == 0)
					dx[j*batchSize + k] = 0;
			}
		}
	}

	return 0;
}

template <typename T>
int DropoutLayer<T>::back_propagation(int t)
{
	int inputStride = inputDim*batchSize;

	// Extract time t variable
	int tidx = t - 1;
	int* D = D_t + inputStride*tidx;
	T* dx = dx_t + inputStride*tidx;

	for (int j = 0; j < inputDim; j++)
	{
		for (int k = 0; k < batchSize; k++)
		{
			// Recall size of D is [hDim, batchSize]
			// Size of dh is [batchSize, hDim]
			if (D[k*inputDim + j] == 0)
				dx[j*batchSize + k] = 0;
		}
	}

	return 0;
}

#pragma endregion DropoutLayer

#pragma region Criterion

class Criterion {
public:
	int periods;
	int batchSize;

	virtual int eval_loss(int t) = 0;
	virtual int eval_dloss(int t) = 0;

	virtual int eval_loss_T()
	{
		int status;
		for (int t = 1; t <= periods; t++)
		{
			status = eval_loss(t);
		}

		return status;
	}

	virtual int eval_dloss_T()
	{
		int status;
		for (int t = 1; t <= periods; t++)
		{
			status = eval_dloss(t);
		}

		return status;
	}
};

template <typename T>
class SoftmaxCriterion : public Criterion {
public:
	int outputDim;

	// Data
	T* y_t_reserve;
	T* y_t;
	// Linear input
	T* ylin_t_reserve;
	T* ylin_t;
	// Logistic transformation
	T* yhat_t;
	// Loss
	T* loss_t;
	// dLoss
	T* dyhat_t;
	// Temp space, used for transpose
	T* temp;

	T temperature;

	SoftmaxCriterion(int _outputDim, int _periods, int _batchSize, T _temperature = 1) :
		outputDim(_outputDim), temperature(_temperature)
	{
		periods = _periods;
		batchSize = _batchSize;

		alloc();
	}

	~SoftmaxCriterion()
	{
		dealloc();
	}

	void alloc();
	void dealloc();

	int eval_loss(int t);
	int eval_dloss(int t);

	void fetch_from_bottom(T* src);
	void fetch_from_bottom_ptr(T* src);
	// Criterion layer, fetch from top is to fetch from output
	void fetch_from_top(T* src);
	void fetch_from_top_ptr(T* src);
};

template<typename T>
void SoftmaxCriterion<T>::fetch_from_top_ptr(T* src)
{
	y_t = src;
}

template<typename T>
void SoftmaxCriterion<T>::fetch_from_top(T* src)
{
	y_t = y_t_reserve;
	memcpy(y_t, src, sizeof(T)*outputDim*batchSize*periods);
}

template<typename T>
void SoftmaxCriterion<T>::fetch_from_bottom_ptr(T* src)
{
	ylin_t = src;
}

template<typename T>
void SoftmaxCriterion<T>::fetch_from_bottom(T* src)
{
	ylin_t = ylin_t_reserve;
	memcpy(ylin_t, src, sizeof(T)*outputDim*batchSize*periods);
}

template <typename T>
void SoftmaxCriterion<T>::alloc()
{
	// Forward
	ylin_t_reserve = new T[outputDim*batchSize*periods];
	ylin_t = ylin_t_reserve;
	y_t_reserve = new T[outputDim*batchSize*periods];
	y_t = y_t_reserve;
	yhat_t = new T[outputDim*batchSize*periods];
	loss_t = new T[batchSize*periods];

	// Back
	dyhat_t = new T[outputDim*batchSize*periods];

	// temp
	temp = new T[outputDim*batchSize];
}

template <typename T>
void SoftmaxCriterion<T>::dealloc()
{
	delete[] y_t_reserve;
	delete[] ylin_t_reserve;
	delete[] yhat_t;
	delete[] loss_t;

	// Back
	delete[] dyhat_t;

	// temp
	delete[] temp;
}

template <typename T>
int SoftmaxCriterion<T>::eval_dloss(int t)
{
	int yStride = batchSize*outputDim;
	// Convert to 0 based
	int tidx = t - 1;

	T* y = y_t + tidx*yStride;
	T* yhat = yhat_t + tidx*yStride;
	T* dyhat = dyhat_t + tidx*yStride;

#pragma simd
	for (int k = 0; k < outputDim*batchSize; k++)
	{
		temp[k] = yhat[k] - y[k];
	}

	// Transpose matrix
	omatcopy('C', 'T', outputDim, batchSize, 1, temp, outputDim, dyhat, batchSize);

	return 0;
}


template <typename T>
int SoftmaxCriterion<T>::eval_loss(int t)
{
	int yStride = batchSize*outputDim;
	// Convert to 0 based
	int tidx = t - 1;

	T* y = y_t + tidx*yStride;
	T* ylin = ylin_t + tidx*yStride;
	T* yhat = yhat_t + tidx*yStride;
	T* loss = loss_t + tidx*batchSize;

	for (int j = 0; j < batchSize; j++)
	{
		T ylin_max = -1e20;
#pragma novector
		for (int k = 0; k < outputDim; k++)
		{
			if (ylin[j*outputDim + k] > ylin_max)
				ylin_max = ylin[j*outputDim + k];
		}

		T sum_exp_ylin_minus_max = 0;

#pragma novector
		for (int k = 0; k < outputDim; k++)
		{
			yhat[j*outputDim + k] = exp((ylin[j*outputDim + k] - ylin_max) / temperature);
			sum_exp_ylin_minus_max += yhat[j*outputDim + k];
		}

		T loss_sum = 0;
#pragma novector
		for (int k = 0; k < outputDim; k++)
		{
			yhat[j*outputDim + k] /= sum_exp_ylin_minus_max;
			loss_sum += -y[j*outputDim + k] * log(yhat[j*outputDim + k]);
		}
		loss[j] = loss_sum;
	}

	return 0;
}

#pragma endregion Criterion


#pragma region Optimizer

template <typename T>
class AdamOptimizer{
public:
	T alpha;
	T beta1;
	T beta2;
	T epsilon;

	int sizeWeights;

	T* m;
	T* v;

	T* dweights;
	T* weights;

	AdamOptimizer(T _alpha, T _beta1, T _beta2, T _epsilon, T _sizeWeights) :
		alpha(_alpha), beta1(_beta1), beta2(_beta2), epsilon(_epsilon), sizeWeights(_sizeWeights)
	{
		m = new T[sizeWeights];
		v = new T[sizeWeights];
		memset(m, 0, sizeof(T)*sizeWeights);
		memset(v, 0, sizeof(T)*sizeWeights);
	}

	~AdamOptimizer()
	{
		delete[] m;
		delete[] v;
	}

	void update_weights(T* weights, T* dweights, int t);
};

template <typename T>
void AdamOptimizer<T>::update_weights(T* weights, T* dweights, int t)
{
	T factor1 = 1 / (1 - pow(beta1, t));
	T factor2 = 1 / (1 - pow(beta2, t));
#pragma simd
	for (int j = 0; j < sizeWeights; j++)
	{
		m[j] *= beta1;
		m[j] += (1 - beta1)*dweights[j];
		v[j] *= beta2;
		v[j] += (1 - beta2)*dweights[j] * dweights[j];
		T mhat = m[j] * factor1;
		T vhat = v[j] * factor2;
		weights[j] -= alpha*mhat / (sqrt(vhat) + epsilon);
	}
}


#pragma endregion Optimizer
