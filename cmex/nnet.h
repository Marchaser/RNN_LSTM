#pragma once

#include "mkl.h"
#include "string.h"
#include "math.h"

#ifdef USE_OMP
#include <omp.h>
#endif

#define SIGMOID(x) 1/(1+exp(-x))

template <typename T>
class LstmLayer {
public:
	int xDim;
	int hDim;
	int periods;
	int xhDim;
	int gifoDim;
	
	// Constructor
	LstmLayer(int _xDim, int _hDim, int _periods)
		: xDim(_xDim), hDim(_hDim), periods(_periods)
	{
		xhDim = xDim + hDim;
		gifoDim = hDim * 4;
	}

	T* weights_x;
	T* weights_h;
	T* biases;

	// Used for forward propagation
	T* x_t;
	T* gifo_lin_t;
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

	void forward_pass(int t, int batchSize);
	void back_propagation(int t, int batchSize);
};

template<>
void LstmLayer<float>::forward_pass(int t, int batchSize)
{
	int xStride = batchSize*xDim;
	int hStride = batchSize*hDim;
	int gifoStride = batchSize*gifoDim;
	// Convert to 0 based
	int tidx = t - 1;

	float* hm;
	float* sm;
	if (t == 1)
	{
		// Use the space for h_t
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
	float* gifo_lin = gifo_lin_t + tidx*gifoStride;
	float* g = g_t + tidx*hStride;
	float* ii = i_t + tidx*hStride;
	float* f = f_t + tidx*hStride;
	float* o = o_t + tidx*hStride;
	float* h = h_t + tidx*hStride;
	float* s = s_t + tidx*hStride;
	float* tanhs = tanhs_t + tidx*hStride;

	// Forward pass one time
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, gifoDim, batchSize, xDim, 1, weights_x, gifoDim, x, xDim, 0, gifo_lin, gifoDim);
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, gifoDim, batchSize, hDim, 1, weights_h, gifoDim, hm, hDim, 1, gifo_lin, gifoDim);

#ifdef USE_OMP
#pragma omp parallel for
#endif
	for (int j = 0; j < batchSize; j++)
	{
		float* gifo_lin_j = &gifo_lin[j*gifoDim];
		float* glin_j = gifo_lin_j + 0 * hDim;
		float* ilin_j = gifo_lin_j + 1 * hDim;
		float* flin_j = gifo_lin_j + 2 * hDim;
		float* olin_j = gifo_lin_j + 3 * hDim;
		float* g_j = g + j*hDim;
		float* ii_j = ii + j*hDim;
		float* f_j = f + j*hDim;
		float* o_j = o + j*hDim;
		float* s_j = s + j*hDim;
		float* tanhs_j = tanhs + j*hDim;
		float* h_j = h + j*hDim;
		float* sm_j = sm + j*hDim;

		// Plus biases
#ifdef USE_OMP
#pragma simd
#endif
		for (int k = 0; k < gifoDim; k++)
		{
			gifo_lin_j[k] += biases[k];
		}

#ifdef USE_OMP
#pragma simd
#endif
		for (int k = 0; k < hDim; k++)
		{
			g_j[k] = tanh(glin_j[k]);
			ii_j[k] = SIGMOID(ilin_j[k]);
			f_j[k] = SIGMOID(flin_j[k]);
			o_j[k] = SIGMOID(olin_j[k]);
			s_j[k] = g_j[k] * ii_j[k] + sm_j[k] * f_j[k];
			tanhs_j[k] = tanh(s_j[k]);
			h_j[k] = tanhs_j[k] * o_j[k];
		}
	}
}

template<>
void LstmLayer<float>::back_propagation(int t, int batchSize)
{
	int xStride = batchSize*xDim;
	int hStride = batchSize*hDim;
	int gifoStride = batchSize*gifoDim;
	// Convert to 0 based
	int tidx = t - 1;

	float* hm;
	float* sm;
	if (t == 1)
	{
		// Use the space for h_t
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
	float* hp = h_t + (tidx + 1)*hStride;
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
#ifdef USE_OMP
#pragma omp parallel for simd
#endif
		for (int j = 0; j < hStride; j++)
		{
			dh[j] = dtop[j];
			ds[j] = 0;
		}
	}
	else
	{
#ifdef USE_OMP
#pragma omp parallel for simd
#endif
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
	float* dglin = dgifo_lin + 0 * hStride;
	float* dilin = dgifo_lin + 1 * hStride;
	float* dflin = dgifo_lin + 2 * hStride;
	float* dolin = dgifo_lin + 3 * hStride;
#ifdef USE_OMP
#pragma omp parallel for simd
#endif
	for (int j = 0; j < hStride; j++)
	{
		doo[j] = dh[j] * tanhs[j];
		dtanhs[j] = dh[j] * o[j];
		ds[j] += dtanhs[j] * (1 - tanhs[j] * tanhs[j]);
		dg[j] = ds[j] * ii[j];
		di[j] = ds[j] * g[j];
		dsm[j] = ds[j] * f[j];
		df[j] = ds[j] * sm[j];

		dglin[j] = dg[j] * (1 - g[j] * g[j]);
		dilin[j] = di[j] * (ii[j] * (1 - ii[j]));
		dflin[j] = df[j] * (f[j] * (1 - f[j]));
		dolin[j] = doo[j] * (o[j] * (1 - o[j]));
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
}


template <typename T>
class SoftmaxLayer {
public: 
	int hDim;
	int yDim;
	int periods;

	// Constructor
	SoftmaxLayer(int _hDim, int _yDim, int _periods) : hDim(_hDim), yDim(_yDim), periods(_periods) {};

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

	void init_dweights()
	{
		memset(dweights, 0, sizeof(T)*hDim*yDim);
		memset(dbiases, 0, sizeof(T)*yDim);
	}

	void forward_pass(int t, int batchSize);

	void back_propagation(int t, int batchSize);
};

template<>
void SoftmaxLayer<float>::forward_pass(int t, int batchSize)
{
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

#ifdef USE_OMP
#pragma omp parallel for
#endif
	for (int j = 0; j < batchSize; j++)
	{
		float* ylin_j = ylin + j*yDim;
		float* yhat_j = yhat + j*yDim;
		float* y_j = y + j*yDim;
		float* loss_j = loss + j;
#ifdef USE_OMP
#pragma simd
#endif
		for (int k = 0; k < yDim; k++)
		{
			ylin_j[k] += biases[k];
		}

		float ylin_max = -1e20;
		for (int k = 0; k < yDim; k++)
		{
			if (ylin_j[k]>ylin_max)
				ylin_max = ylin_j[k];
		}

		float sum_exp_ylin_minus_max = 0;
#ifdef USE_OMP
#pragma simd reduction(+:sum_exp_ylin_minus_max)
#endif
		for (int k = 0; k < yDim; k++)
		{
			yhat_j[k] = exp(ylin_j[k] - ylin_max);
			sum_exp_ylin_minus_max += yhat_j[k];
		}

		*loss_j = 0;
#ifdef USE_OMP
#pragma simd reduction(+:(*loss_j))
#endif
		for (int k = 0; k < yDim; k++)
		{
			yhat_j[k] /= sum_exp_ylin_minus_max;
			(*loss_j) += -y_j[k] * log(yhat_j[k]);
		}
	}
}

template<>
void SoftmaxLayer<float>::back_propagation(int t, int batchSize)
{
	int yStride = batchSize*yDim;
	int hStride = batchSize*hDim;
	// Convert to 0 based
	int tidx = t - 1;

	float* h = h_t + tidx*hStride;
	float* y = y_t + tidx*yStride;
	float* ylin = ylin_t + tidx*yStride;
	float* yhat = yhat_t + tidx*yStride;
	float* dh = dh_t + tidx*hStride;

#ifdef USE_OMP
#pragma omp parallel for simd
#endif
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
}
