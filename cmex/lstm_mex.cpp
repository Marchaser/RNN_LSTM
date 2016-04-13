// Recurrent Neural Network
// Author: Wenlan Luo (luowenlan at gmail.com), Georgetown University
// lstm_mex.cpp
// Last updated: 2016-4-11

#include <signal.h>

#include "mex.h"
#include "math.h"
#include "MatlabMatrix.h"
#include <string.h>
#include "nnet.h"

#ifdef USE_OMP
#include <omp.h>
#endif

#define SIGMOID(x) 1/(1+exp(-x))

void my_function_to_handle_aborts(int signal_number)
{
	/*
	printf("Break here\n");
	exit(-1);
	*/
	char ErrMsg[200];
	sprintf(ErrMsg, "Abort from CMEX.\nLINE: %d\nFILE: %s\n", __LINE__, __FILE__);
	mexErrMsgTxt(ErrMsg);
}

#define CRRA(c) ( pow((c),1-Sigma)/(1-Sigma) )

void TRAIN();
void TRAIN2();
void update_der(float* W, float* dW, int nW, float learningRate);


using namespace MatlabMatrix;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Handle errors
	signal(SIGABRT, &my_function_to_handle_aborts);

	// Get task number
	GET_INT(MEX_TASK);
	GET_INT(MEX_TRAIN);

	if (MEX_TASK == MEX_TRAIN)
	{
		TRAIN2();
	}
}

void TRAIN2()
{
	// Get parameters from workspace
	GET_SGL(temperature);
	GET_INT(batchSize);
	GET_SGL(learningRate);
	GET_INT(T);
	GET_INT(saveFreq);
	GET_INT(hDim);
	GET_INT(xDim);
	GET_INT(yDim);
	GET_INT(Ts);
	GET_INT(numDropOuts);

	// Get Data
	GET_FM(xData, 3);
	GET_FM(yData, 3);

	// Get input
	GET_FM(W_gifo_x, 2);
	GET_FM(W_gifo_h, 2);
	GET_FM(b_gifo, 2);
	GET_FM(Wyh, 2);
	GET_FM(by, 2);

	// Get derivative
	GET_FM(dW_gifo_x, 2);
	GET_FM(dW_gifo_h, 2);
	GET_FM(db_gifo, 2);
	GET_FM(dWyh, 2);
	GET_FM(dby, 2);

	// Initate computation
	GET_FM(gifo_t, 3);
	GET_FM(gifo_lin_t, 3);
	GET_FM(g_t, 3);
	GET_FM(i_t, 3);
	GET_FM(f_t, 3);
	GET_FM(o_t, 3);
	GET_FM(h_t, 3);
	GET_FM(hD_t, 3);
	GET_FM(s_t, 3);
	GET_FM(tanhs_t, 3);
	GET_FM(ylin_t, 3);
	GET_FM(yhat_t, 3);
	GET_FM(ZEROS, 2);
	GET_FM(ONESBatchSize, 2);

	GET_FM(dh, 2);
	GET_FM(ds, 2);
	GET_FM(dyhat, 2);
	GET_FM(doo, 2);
	GET_FM(dtanhs, 2);
	GET_FM(dg, 2);
	GET_FM(di, 2);
	GET_FM(df, 2);
	GET_FM(dyhat_temp, 2);
	GET_FM(g_temp, 2);
	GET_FM(dgifo_lin, 2);

	GET_FM(dhm, 2);
	GET_FM(dsm, 2);
	GET_FM(dx_t, 3);
	GET_FM(dh_t, 3);
	GET_FM(loss_t, 2);

	GET_INT(NumThreads);
	mkl_set_num_threads(NumThreads);
#ifdef USE_OMP
	omp_set_num_threads(NumThreads);
#endif

	// Construct network
	LstmLayer<float> lstmLayer1(xDim, hDim, T);
	SoftmaxLayer<float> softmaxLayer(hDim, yDim, T);

	// Assign space to LSTM network
	lstmLayer1.weights_x = _W_gifo_x;
	lstmLayer1.weights_h = _W_gifo_h;
	lstmLayer1.biases = _b_gifo;
	lstmLayer1.dweights_x = _dW_gifo_x;
	lstmLayer1.dweights_h = _dW_gifo_h;
	lstmLayer1.dbiases = _db_gifo;

	lstmLayer1.x_t = _xData;
	lstmLayer1.gifo_lin_t = _gifo_lin_t;
	lstmLayer1.g_t = _g_t;
	lstmLayer1.i_t = _i_t;
	lstmLayer1.f_t = _f_t;
	lstmLayer1.o_t = _o_t;
	lstmLayer1.h_t = _h_t;
	lstmLayer1.s_t = _s_t;
	lstmLayer1.tanhs_t = _tanhs_t;

	// Information within current period
	lstmLayer1.dh = _dh;
	lstmLayer1.ds = _ds;
	lstmLayer1.doo = _doo;
	lstmLayer1.dtanhs = _dtanhs;
	lstmLayer1.dg = _dg;
	lstmLayer1.di = _di;
	lstmLayer1.df = _df;
	lstmLayer1.dgifo_lin = _dgifo_lin;
	// Information down to time period
	lstmLayer1.dhm = _dhm;
	lstmLayer1.dsm = _dsm;
	// Information down to the lower layer
	lstmLayer1.dx_t = _dx_t;

	// Temporary space
	lstmLayer1.g_temp = _g_temp;

	// Constant
	lstmLayer1.ZEROS_H_STRIDE = _ZEROS;
	lstmLayer1.ONES_BATCH_SIZE = _ONESBatchSize;

	// Assign space to Softmax network
	softmaxLayer.weights = _Wyh;
	softmaxLayer.biases = _by;
	softmaxLayer.dweights = _dWyh;
	softmaxLayer.dbiases = _dby;

	softmaxLayer.y_t = _yData;
	softmaxLayer.ylin_t = _ylin_t;
	softmaxLayer.yhat_t = _yhat_t;
	softmaxLayer.loss_t = _loss_t;
	softmaxLayer.dyhat = _dyhat;
	softmaxLayer.dh_t = _dh_t;
	softmaxLayer.ONES_BATCH_SIZE = _ONESBatchSize;

	// Network connections
	softmaxLayer.h_t = lstmLayer1.h_t;
	lstmLayer1.dtop_t = softmaxLayer.dh_t;

	// Initiate weights
	lstmLayer1.init_dweights();
	softmaxLayer.init_dweights();

	// Train once
	for (int t = 1; t <= T ; t++)
	{
		// forward
		lstmLayer1.forward_pass(t, batchSize);
		softmaxLayer.forward_pass(t, batchSize);
	}

	PUT(yhat_t);
	PUT(h_t);

	for (int t = T; t >= 1 ; t--)
	{
		// backward
		softmaxLayer.back_propagation(t, batchSize);
		lstmLayer1.back_propagation(t, batchSize);
	}

	PUT(loss_t);
	PUT(dW_gifo_x);
	PUT(dW_gifo_h);
	PUT(db_gifo);
	PUT(dWyh);
	PUT(dby);
}

/*
void TRAIN()
{
	// Get parameters from workspace
	GET_SGL(temperature);
	GET_INT(batchSize);
	GET_SGL(learningRate);
	GET_INT(T);
	GET_INT(saveFreq);
	GET_INT(gDim);
	GET_INT(WDim);
	GET_INT(xDim);
	GET_INT(yDim);
	GET_INT(Ts);
	GET_INT(numDropOuts);

	// Get Data
	GET_FM(xData, 2);
	GET_FM(yData, 2);

	// Get input
	GET_FM(W_gifo_x, 2);
	GET_FM(W_gifo_h, 2);
	GET_FM(b_gifo, 2);
	GET_FM(Wyh, 2);
	GET_FM(by, 2);

	// Get derivative
	GET_FM(dW_gifo_x, 2);
	GET_FM(dW_gifo_h, 2);
	GET_FM(db_gifo, 2);
	GET_FM(dWyh, 2);
	GET_FM(dby, 2);

	// Initate computation
	GET_FM(gifo_t, 3);
	GET_FM(gifo_lin_t, 3);
	GET_FM(g_t, 3);
	GET_FM(i_t, 3);
	GET_FM(f_t, 3);
	GET_FM(o_t, 3);
	GET_FM(h_t, 3);
	GET_FM(hD_t, 3);
	GET_FM(s_t, 3);
	GET_FM(tanhs_t, 3);
	GET_FM(ylin_t, 3);
	GET_FM(yhat_t, 3);
	GET_FM(ZEROS, 2);
	GET_FM(ONESBatchSize, 2);

	GET_FM(dh, 2);
	GET_FM(dhD, 2);
	GET_FM(ds, 2);
	GET_FM(dyhat, 2);
	GET_FM(doo, 2);
	GET_FM(dtanhs, 2);
	GET_FM(dg, 2);
	GET_FM(di, 2);
	GET_FM(df, 2);
	GET_FM(dyhat_temp, 2);
	GET_FM(g_temp, 2);
	GET_FM(dgifo_lin, 2);
	float* dglin = _dgifo_lin + 0 * gDim*batchSize;
	float* dilin = _dgifo_lin + 1 * gDim*batchSize;
	float* dflin = _dgifo_lin + 2 * gDim*batchSize;
	float* dolin = _dgifo_lin + 3 * gDim*batchSize;

	// Dropout
	GET_FM(dropOutDraws_all, 2);

	// Precompute gifo_x
	GET_FM(gifo_x_t, 2);

	// Set computation environment
	GET_INT(NumThreads);
	mkl_set_num_threads(NumThreads);
#ifdef USE_OMP
	omp_set_num_threads(NumThreads);
#endif

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, WDim, T - 1, xDim, 1, _W_gifo_x, WDim, _xData, xDim, 0, &gifo_x_t(1, batchSize + 1), WDim);

	int batchStart = 1;
	int pNewDataStart = T;

	int dropOutCount = 0;

	int saveCount = 0;

	while (batchStart <= Ts)
	{
		int batchEnd = batchStart + batchSize - 1;
		int pDataEnd = batchEnd + T - 1;

		if (pDataEnd > Ts)
			break;

		// Compute gifo_x
		memcpy(_gifo_x_t, &gifo_x_t(1, batchSize + 1), sizeof(float)*WDim*(T - 1));
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, WDim, pDataEnd - pNewDataStart + 1, xDim, 1, _W_gifo_x, WDim, &xData(1, pNewDataStart), xDim, 0, &gifo_x_t(1, T), WDim);

		// Extract x, y
		// Use blitz to create memory view
		float* x_batch = &xData(1, batchStart);
		float* y_batch = &yData(1, batchStart);
		// FM(x, &xData(1, batchStart), 2, xDim, pDataEnd - batchStart + 1);
		// FM(y, &yData(1, batchStart), 2, yDim, pDataEnd - batchStart + 1);

		// Shift pointer
		batchStart = batchEnd + 1;
		pNewDataStart = pDataEnd + 1;

		// Create memory view of dropouts
		// Reuse random numbers
		if (dropOutCount + gDim*batchSize*T >= gDim*numDropOuts*T)
			dropOutCount = 0;
		FM(dropOutDraws_t, _dropOutDraws_all + dropOutCount, 3, gDim, batchSize, T);
		dropOutCount += gDim*batchSize*T;

		// Forward pass
		for (int t = 1; t <= T; t++)
		{
			float* hm;
			float* sm;
			if (t == 1)
			{
				hm = ZEROS.data();
				sm = ZEROS.data();
			}
			else
			{
				hm = &h_t(1, 1, t - 1);
				sm = &s_t(1, 1, t - 1);
			}

			// Extract time t variable
			float* gifo_lin = &gifo_lin_t(1, 1, t);
			float* gifo_x = &gifo_x_t(1, t);
			float* g = &g_t(1, 1, t);
			float* ii = &i_t(1, 1, t);
			float* f = &f_t(1, 1, t);
			float* o = &o_t(1, 1, t);
			float* h = &h_t(1, 1, t);
			float* hD = &hD_t(1, 1, t);
			float* s = &s_t(1, 1, t);
			float* tanhs = &tanhs_t(1, 1, t);
			float* ylin = &ylin_t(1, 1, t);
			float* yhat = &yhat_t(1, 1, t);

			// Create memory view of dropOuts
			float* dropOutDraws = &dropOutDraws_t(1, 1, t);

			// Forward pass one time
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, WDim, batchSize, gDim, 1, _W_gifo_h, WDim, hm, gDim, 0, gifo_lin, WDim);
#ifdef USE_OMP
#pragma omp parallel for simd
#endif
			for (int i = 0; i < WDim*batchSize; i++)
			{
				gifo_lin[i] += gifo_x[i];
			}

#ifdef USE_OMP
#pragma omp parallel for
#endif
			for (int j = 0; j < batchSize; j++)
			{
				float* gifo_lin_j = &gifo_lin[j*WDim];
				float* glin_j = gifo_lin_j + 0 * gDim;
				float* ilin_j = gifo_lin_j + 1 * gDim;
				float* flin_j = gifo_lin_j + 2 * gDim;
				float* olin_j = gifo_lin_j + 3 * gDim;
				float* g_j = g + j*gDim;
				float* ii_j = ii + j*gDim;
				float* f_j = f + j*gDim;
				float* o_j = o + j*gDim;
				float* s_j = s + j*gDim;
				float* tanhs_j = tanhs + j*gDim;
				float* h_j = h + j*gDim;
				float* hD_j = hD + j*gDim;
				float* sm_j = sm + j*gDim;
				float* dropOutDraws_j = dropOutDraws + j*gDim;

#ifdef USE_OMP
#pragma simd
#endif
				for (int k = 0; k < WDim; k++)
				{
					gifo_lin_j[k] += _b_gifo[k];
				}

#ifdef USE_OMP
#pragma simd
#endif
				for (int k = 0; k < gDim; k++)
				{
					g_j[k] = tanh(glin_j[k]);
					ii_j[k] = SIGMOID(ilin_j[k]);
					f_j[k] = SIGMOID(flin_j[k]);
					o_j[k] = SIGMOID(olin_j[k]);
					s_j[k] = g_j[k] * ii_j[k] + sm_j[k] * f_j[k];
					tanhs_j[k] = tanh(s_j[k]);
					h_j[k] = tanhs_j[k] * o_j[k];
					hD_j[k] = h_j[k] * dropOutDraws_j[k];
				}
			}

			// Layer output to final output
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, yDim, batchSize, gDim, 1, _Wyh, yDim, hD, gDim, 0, ylin, yDim);
#ifdef USE_OMP
#pragma omp parallel for
#endif
			for (int j = 0; j < batchSize; j++)
			{
				float* ylin_j = ylin + j*yDim;
				float* yhat_j = yhat + j*yDim;
#ifdef USE_OMP
#pragma simd
#endif
				for (int k = 0; k < yDim; k++)
				{
					ylin_j[k] += _by[k];
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
#ifdef USE_OMP
#pragma simd
#endif
				for (int k = 0; k < yDim; k++)
				{
					yhat_j[k] /= sum_exp_ylin_minus_max;
				}
			}
		}

		// Backward propagation
		zeros(dW_gifo_x);
		zeros(dW_gifo_h);
		zeros(db_gifo);
		zeros(dWyh);
		zeros(dby);
		zeros(dh);
		zeros(ds);
		for (int t = T; t >= 1; t--)
		{
			float* hm;
			float* sm;
			if (t == 1)
			{
				hm = ZEROS.data();
				sm = ZEROS.data();
			}
			else
			{
				hm = &h_t(1, 1, t - 1);
				sm = &s_t(1, 1, t - 1);
			}

			// Extract time t variable
			float* g = &g_t(1, 1, t);
			float* ii = &i_t(1, 1, t);
			float* f = &f_t(1, 1, t);
			float* o = &o_t(1, 1, t);
			float* h = &h_t(1, 1, t);
			float* hD = &h_t(1, 1, t);
			float* tanhs = &tanhs_t(1, 1, t);
			float* ylin = &ylin_t(1, 1, t);
			float* yhat = &yhat_t(1, 1, t);
			float* y = y_batch + (t - 1)*yDim;
			float* x = x_batch + (t - 1)*xDim;

			// Create memory view of dropOuts
			float* dropOutDraws = &dropOutDraws_t(1, 1, t);

			// Back propagation for one time step
#ifdef USE_OMP
#pragma omp parallel for simd
#endif
			for (int k = 0; k < yDim*batchSize; k++)
			{
				_dyhat_temp[k] = yhat[k] - y[k];
			}
			// Transpose matrix
			mkl_somatcopy('C', 'T', yDim, batchSize, 1, _dyhat_temp, yDim, _dyhat, batchSize);

			// Transpose all relevant matrix
			mkl_somatcopy('C', 'T', gDim, batchSize, 1, g, gDim, _g_temp, batchSize);
			memcpy(g, _g_temp, sizeof(float)*gDim*batchSize);

			mkl_somatcopy('C', 'T', gDim, batchSize, 1, ii, gDim, _g_temp, batchSize);
			memcpy(ii, _g_temp, sizeof(float)*gDim*batchSize);

			mkl_somatcopy('C', 'T', gDim, batchSize, 1, f, gDim, _g_temp, batchSize);
			memcpy(f, _g_temp, sizeof(float)*gDim*batchSize);

			mkl_somatcopy('C', 'T', gDim, batchSize, 1, o, gDim, _g_temp, batchSize);
			memcpy(o, _g_temp, sizeof(float)*gDim*batchSize);

			mkl_somatcopy('C', 'T', gDim, batchSize, 1, sm, gDim, _g_temp, batchSize);
			memcpy(sm, _g_temp, sizeof(float)*gDim*batchSize);

			mkl_somatcopy('C', 'T', gDim, batchSize, 1, tanhs, gDim, _g_temp, batchSize);
			memcpy(tanhs, _g_temp, sizeof(float)*gDim*batchSize);

			// dWyh = dWyh + dyhat'*hD';
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, yDim, gDim, batchSize, 1, _dyhat_temp, yDim, hD, gDim, 1, _dWyh, yDim);

			// dby = dby + sum(dyhat,1)';
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, yDim, 1, batchSize, 1, _dyhat_temp, yDim, _ONESBatchSize, batchSize, 1, _dby, yDim);

			// dh = dh + (dyhat * Wyh).*dropOutDraws'
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, batchSize, gDim, yDim, 1, _dyhat, batchSize, _Wyh, yDim, 0, _dhD, batchSize);
#ifdef USE_OMP
#pragma omp parallel for simd
#endif
			for (int j = 0; j < batchSize*gDim; j++)
			{
				_dh[j] += _dhD[j] * dropOutDraws[j];
			}

			// do = dh.*tanhs';
			// dtanhs = dh.*o';
			// ds = ds + dtanhs.*(1-tanhs.^2)';
			// dg = ds.*ii';
			// di = ds.*g';
			// df = ds.*sm';
			// dsm = ds.*f';
			// dglin = dg.*(1-g.^2)';
			// dilin = di.*(ii.*(1-ii))';
			// dflin = df.*(f.*(1-f))';
			// dolin = do.*(o.*(1-o))';
			// dgifo_lin = [dglin,dilin,dflin,dolin];
#ifdef USE_OMP
#pragma omp parallel for simd
#endif
			for (int j = 0; j < batchSize*gDim; j++)
			{
				_doo[j] = _dh[j] * tanhs[j];
				_dtanhs[j] = _dh[j] * o[j];
				_ds[j] += _dtanhs[j] * (1 - tanhs[j] * tanhs[j]);
				_dg[j] = _ds[j] * ii[j];
				_di[j] = _ds[j] * g[j];
				_df[j] = _ds[j] * sm[j];

				_ds[j] = _ds[j] * f[j];

				dglin[j] = _dg[j] * (1 - g[j] * g[j]);
				dilin[j] = _di[j] * (ii[j] * (1 - ii[j]));
				dflin[j] = _df[j] * (f[j] * (1 - f[j]));
				dolin[j] = _doo[j] * (o[j] * (1 - o[j]));
			}

			// dW_gifo_x = dW_gifo_x + dgifo_lin'*x(:,t:t+batchSize-1)';
			cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans, WDim, xDim, batchSize, 1, _dgifo_lin, batchSize, x, xDim, 1, _dW_gifo_x, WDim);
			// dW_gifo_h = dW_gifo_h + dgifo_lin'*hm';
			cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans, WDim, gDim, batchSize, 1, _dgifo_lin, batchSize, hm, gDim, 1, _dW_gifo_h, WDim);
			// db_gifo = db_gifo + sum(dgifo_lin,1)';
			cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, WDim, 1, batchSize, 1, _dgifo_lin, batchSize, _ONESBatchSize, batchSize, 1, _db_gifo, WDim);

			// dh = dgifo_lin * W_gifo_h;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, batchSize, gDim, WDim, 1, _dgifo_lin, batchSize, _W_gifo_h, WDim, 0, _dh, batchSize);
			}

		// Update derivative
		update_der(_W_gifo_x, _dW_gifo_x, WDim*xDim, learningRate);
		update_der(_W_gifo_h, _dW_gifo_h, WDim*gDim, learningRate);
		update_der(_b_gifo, _db_gifo, WDim * 1, learningRate);
		update_der(_Wyh, _dWyh, yDim*gDim, learningRate);
		update_der(_by, _dby, yDim * 1, learningRate);

		saveCount++;

		if (saveCount % saveFreq == 0)
		{
			// Save variable in work space
			PUT(W_gifo_x);
			PUT(W_gifo_h);
			PUT(b_gifo);
			PUT(Wyh);
			PUT(by);

			PUT_SCALAR(saveCount);
			PUT_SCALAR(batchEnd);

			// Call the output script
			mexCallMATLAB(0, 0, 0, 0, "output_func");
		}
		}
	PUT(W_gifo_x);
	PUT(W_gifo_h);
	PUT(b_gifo);
	PUT(Wyh);
	PUT(by);
	}

inline void update_der(float* W, float* dW, int nW, float learningRate)
{
#ifdef USE_OMP
#pragma omp parallel for simd
#endif
	for (int j = 0; j < nW; j++)
	{
		W[j] -= learningRate*dW[j];
	}
}
*/

