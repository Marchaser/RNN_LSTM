// Recurrent Neural Network
// Author: Wenlan Luo (luowenlan at gmail.com), Georgetown University
// lstm_mex.cpp
// Last updated: 2016-4-11

#include <signal.h>

#include "mex.h"
#include "math.h"
#include "MatlabMatrix.h"
#include <string.h>

#ifndef TYPENAME_FLOAT
#define TYPENAME_DOUBLE
#endif

#include "nnet.h"

#if defined(TYPENAME_DOUBLE)
#define TYPENAME double
#define GET_V_VIEW GET_DV_VIEW
#define GET_SCALAR GET_DBL
#elif defined(TYPENAME_FLOAT)
#define TYPENAME float
#define GET_V_VIEW GET_FV_VIEW
#define GET_SCALAR GET_SGL
#endif

#define MAX_LAYER 4
#define MAX_THREAD 8

#ifdef USE_OMP
#include <omp.h>
#endif

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

void ALLOC();
void INFO();
void TRAIN();
void PREDICT();
void PREDICT_BATCH();
void INIT_SEED();
void INIT_MEMORY();
void PRE_TREAT();

using namespace MatlabMatrix;


// Some variable to persist between calls
// Space, thread specific
LstmLayer<TYPENAME>* lstm_thread[MAX_LAYER][MAX_THREAD];
DropoutLayer<TYPENAME>* dropout_thread[MAX_LAYER][MAX_THREAD];
LinearLayer<TYPENAME>* linear_thread[MAX_THREAD];
SoftmaxCriterion<TYPENAME>* softmax_thread[MAX_THREAD];
AdamOptimizer<TYPENAME>* optimizer;

void ExitFcn()
{
	for (int thread_id = 0; thread_id < MAX_THREAD; thread_id++)
	{
		for (int i = 0; i < MAX_LAYER; i++)
		{
			if (lstm_thread[i][thread_id] != 0)
				delete lstm_thread[i][thread_id];
			if (dropout_thread[i][thread_id] != 0)
				delete dropout_thread[i][thread_id];
		}
		if (linear_thread[thread_id] != 0)
			delete linear_thread[thread_id];
		if (softmax_thread[thread_id]!=0)
			delete softmax_thread[thread_id];

	}
	// Optimizer
	delete optimizer;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Handle errors
	signal(SIGABRT, &my_function_to_handle_aborts);

	// Register clear function
	mexAtExit(ExitFcn);

	mkl_set_num_threads(1);

	// Allocate space only once
	static int isAlloc = 0;
	if (isAlloc == 0)
	{
		ALLOC();
		isAlloc = 1;
	}

	// Get task number
	GET_INT(MEX_TASK);
	GET_INT(MEX_ALLOC);
	GET_INT(MEX_INFO);
	GET_INT(MEX_INIT_SEED);
	GET_INT(MEX_INIT_MEMORY);
	GET_INT(MEX_PRE_TREAT);
	GET_INT(MEX_TRAIN);
	GET_INT(MEX_PREDICT);
	GET_INT(MEX_PREDICT_BATCH);

	if (MEX_TASK == MEX_ALLOC & isAlloc == 0)
	{
		ALLOC();
		isAlloc = 1;
		return;
	}

	if (MEX_TASK == MEX_INFO)
	{
		INFO();
		return;
	}

	if (MEX_TASK == MEX_INIT_SEED)
	{
		INIT_SEED();
		return;
	}

	if (MEX_TASK == MEX_INIT_MEMORY)
	{
		INIT_MEMORY();
		return;
	}

	if (MEX_TASK == MEX_PRE_TREAT)
	{
		PRE_TREAT();
		return;
	}

	if (MEX_TASK == MEX_TRAIN)
	{
		TRAIN();
		return;
	}

	if (MEX_TASK == MEX_PREDICT)
	{
		PREDICT();
		return;
	}

	if (MEX_TASK == MEX_PREDICT_BATCH)
	{
		PREDICT_BATCH();
		return;
	}

	mexErrMsgTxt("No Task executed");
}

void INFO()
{
	GET_INT(nLayer);

	// Put relevant nnet info to workspace
	int sizeWeights = 0;
	for (int i = 0; i < nLayer; i++)
	{
		sizeWeights += lstm_thread[i][0]->sizeWeights;
	}
	sizeWeights += linear_thread[0]->sizeWeights;

	PUT_SCALAR(sizeWeights);
}

void ALLOC()
{
	GET_INT(xDim);
	GET_IV_VIEW(hDims);
	GET_INT(yDim);
	GET_INT(periods);
	GET_INT(nLayer);
	GET_INT(batchSizeThread);
	GET_INT(NumThreads);
	GET_DBL(dropoutRate);
	GET_INT(dropoutSeed);

	if (nLayer > MAX_LAYER)
	{
		mexErrMsgTxt("nLayer > MAX_LAYER, recompile with redefining MAX_LAYER");
	}

	if (NumThreads > MAX_THREAD)
	{
		mexErrMsgTxt("NumThreads > MAX_THREAD, recompile with redefining MAX_THREAD");
	}

	for (int thread_id = 0; thread_id < NumThreads; thread_id++)
	{
		// First Layer
		if (lstm_thread[0][thread_id] == 0)
			lstm_thread[0][thread_id] = new LstmLayer<TYPENAME>(xDim, _hDims[0], periods, batchSizeThread);
		if (dropout_thread[0][thread_id] == 0)
			dropout_thread[0][thread_id] = new DropoutLayer<TYPENAME>(_hDims[0], periods, batchSizeThread);
		// All future Layers
		for (int i = 1; i < nLayer; i++)
		{
			if (lstm_thread[i][thread_id] == 0)
				lstm_thread[i][thread_id] = new LstmLayer<TYPENAME>(_hDims[i - 1], _hDims[i], periods, batchSizeThread);
			if (dropout_thread[i][thread_id] == 0)
				dropout_thread[i][thread_id] = new DropoutLayer<TYPENAME>(_hDims[i], periods, batchSizeThread);
		}
		// Last Layer
		if (linear_thread[thread_id] == 0)
			linear_thread[thread_id] = new LinearLayer<TYPENAME>(_hDims[nLayer-1], yDim, periods, batchSizeThread);
		if (softmax_thread[thread_id] == 0)
			softmax_thread[thread_id] = new SoftmaxCriterion<TYPENAME>(yDim, periods, batchSizeThread);
	}

	// Put relevant nnet info to workspace
	int sizeWeights = 0;
	for (int i = 0; i < nLayer; i++)
	{
		sizeWeights += lstm_thread[i][0]->sizeWeights;
	}
	sizeWeights += linear_thread[0]->sizeWeights;

	PUT_SCALAR(sizeWeights);

	GET_SCALAR(learningRate);
	GET_SCALAR(Adam_beta1);
	GET_SCALAR(Adam_beta2);
	GET_SCALAR(Adam_epsilon);
	optimizer = new AdamOptimizer<TYPENAME>(learningRate, Adam_beta1, Adam_beta2, Adam_epsilon, sizeWeights);

	INIT_SEED();
	INIT_MEMORY();
}

void PRE_TREAT()
{
	GET_INT(xDim);
	GET_INT(nLayer);
	GET_IV_VIEW(hDims);
	GET_INT(yDim);
	GET_INT(periods);
	GET_INT(batchSizeThread);

	GET_V_VIEW(weights);


	// Assign space to network
	TYPENAME* ptr = _weights;

	for (int i = 0; i < nLayer ; i++)
	{
		lstm_thread[i][0]->assign_weights(&ptr);

		// Encourage memory at the beginning
		TYPENAME* ptr_f_biases = lstm_thread[i][0]->get_f_biases();
		for (int j = 0; j < _hDims[i]; j++)
		{
			ptr_f_biases[j] = 1;
		}
	}
	linear_thread[0]->assign_weights(&ptr);
}

void INIT_SEED()
{
	// This function should allocate space for layers, and return size of weights
	GET_INT(nLayer);
	GET_INT(NumThreads);
	GET_DBL(dropoutRate);
	GET_INT(dropoutSeed);

	// Initiate something
	for (int thread_id = 0; thread_id < NumThreads; thread_id++)
	{
		for (int i = 0; i < nLayer; i++)
		{
			// initiate dropout
			dropout_thread[i][thread_id]->seed = dropoutSeed + i + thread_id;
			dropout_thread[i][thread_id]->dropoutRate = dropoutRate;
		}
	}
}

void INIT_MEMORY()
{
	// This function should allocate space for layers, and return size of weights
	GET_INT(nLayer);
	GET_INT(NumThreads);
	GET_DBL(dropoutRate);
	GET_INT(dropoutSeed);

	// Initiate something
	for (int thread_id = 0; thread_id < NumThreads; thread_id++)
	{
		for (int i = 0; i < nLayer; i++)
		{
			// Information is truncated
			lstm_thread[i][thread_id]->clear_info();
		}
	}
}

void PREDICT_BATCH()
{
	// Get parameters from workspace
	GET_INT(batchSizeThread);
	GET_INT(periods);
	GET_INT(xDim);
	GET_INT(nLayer);
	GET_IV_VIEW(hDims);
	GET_INT(yDim);
	GET_INT(NumThreads);
	GET_INT(sizeWeights);
	GET_DBL(dropoutRate);
	GET_INT(step);
	GET_SCALAR(temperature);

	// Get Data
	GET_DV_VIEW(batchDataStride);
	GET_V_VIEW(xData_t);
	GET_V_VIEW(yData_t);

	// Get input
	GET_V_VIEW(weights);

	// Get loss function
	GET_V_VIEW(loss_thread);

#ifdef USE_OMP
	omp_set_num_threads(NumThreads);
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
	for (int thread_id = 0; thread_id < NumThreads; thread_id++)
	{
		// Get thread local network
		LstmLayer<TYPENAME>* lstm[MAX_LAYER];
		TYPENAME* ptr_weights = _weights;
		for (int i = 0; i < nLayer; i++)
		{
			lstm[i] = lstm_thread[i][thread_id];
			lstm[i]->assign_weights(&ptr_weights);
		}
		LinearLayer<TYPENAME>* linear = linear_thread[thread_id];
		linear->assign_weights(&ptr_weights);
		
		SoftmaxCriterion<TYPENAME>* softmax = softmax_thread[thread_id];
		softmax->temperature = temperature;

		// Get thread local output
		TYPENAME* loss = _loss_thread + thread_id*periods*batchSizeThread;

		// Copy data
		for (int j = 0; j < batchSizeThread; j++)
		{
			int startingPos = (int)_batchDataStride[thread_id*batchSizeThread + j];
			for (int t = 0; t < periods; t++)
			{
				memcpy(lstm[0]->x_t + t*batchSizeThread*xDim + j*xDim, _xData_t + xDim*startingPos + xDim*t, sizeof(TYPENAME)*xDim);
				memcpy(softmax->y_t + t*batchSizeThread*yDim + j*yDim, _yData_t + xDim*startingPos + yDim*t, sizeof(TYPENAME)*yDim);
			}
		}

		int status;

		// Train
		// Forward
		// First Layer
		status = lstm[0]->forward_pass_T();
		// Future Layer
		for (int i = 1; i < nLayer; i++)
		{
			lstm[i]->fetch_from_bottom_with_dropout(lstm[i - 1]->h_t, dropoutRate);
			status = lstm[i]->forward_pass_T();
		}
		// Last Layer to Softmax
		linear->fetch_from_bottom_with_dropout(lstm[nLayer - 1]->h_t, dropoutRate);
		status = linear->forward_pass_T();
		softmax->fetch_from_bottom_ptr(linear->y_t);
		status = softmax->eval_loss_T();

		// Output loss
		memcpy(loss, softmax->loss_t, sizeof(TYPENAME)*batchSizeThread*periods);

		// Store information for next training
		for (int i = 0; i < nLayer ; i++)
		{
			lstm[i]->store_info(lstm[i]->h_t, lstm[i]->s_t);
		}
	}
}

void PREDICT()
{
	// Parameters
	GET_INT(batchSize);
	GET_INT(periods);
	GET_INT(xDim);
	GET_INT(nLayer);
	GET_IV_VIEW(hDims);
	GET_INT(yDim);
	GET_INT(NumThreads);
	GET_SCALAR(dropoutRate);
	GET_SCALAR(temperature);

	mkl_set_num_threads(NumThreads);

	// Output
	GET_V_VIEW(yhat_t);

	// Get Data
	GET_V_VIEW(xData);
	GET_V_VIEW(yData);
	GET_V_VIEW(weights);

	// Thread level layer
	LstmLayer<TYPENAME>* lstm[MAX_LAYER];
	for (int i = 0; i < nLayer ; i++)
	{
		lstm[i] = lstm_thread[i][0];
	}
	LinearLayer<TYPENAME>* linear = linear_thread[0];
	SoftmaxCriterion<TYPENAME>* softmax = softmax_thread[0];
	softmax->temperature = temperature;

	// Networks for evaluation

	// Assign space to network
	TYPENAME* ptr = _weights;
	for (int i = 0; i < nLayer ; i++)
	{
		lstm[i]->assign_weights(&ptr);
	}
	linear->assign_weights(&ptr);

	// Copy data
	lstm[0]->fetch_from_bottom_ptr(_xData);
	softmax->fetch_from_top_ptr(_yData);

	int status;
	// Forward
	// First Layer
	lstm[0]->forward_pass_T();
	// Future Layer
	for (int i = 1; i < nLayer ; i++)
	{
		lstm[i]->fetch_from_bottom_with_dropout(lstm[i - 1]->h_t, dropoutRate);
		lstm[i]->forward_pass_T();
	}
	// Last Layer to Softmax
	linear->fetch_from_bottom_with_dropout(lstm[nLayer - 1]->h_t, dropoutRate);
	linear->forward_pass_T();

	softmax->fetch_from_bottom_ptr(linear->y_t);
	softmax->eval_loss_T();

	// Store information
	for (int i = 0; i < nLayer ; i++)
	{
		lstm[i]->store_info(lstm[i]->h_t, lstm[i]->s_t);
	}

	// Copy to output
	memcpy(_yhat_t, softmax->yhat_t, sizeof(TYPENAME)*yDim*batchSize*periods);
}

void TRAIN()
{
	// Get parameters from workspace
	GET_INT(batchSizeThread);
	GET_INT(periods);
	GET_INT(xDim);
	GET_INT(nLayer);
	GET_IV_VIEW(hDims);
	GET_INT(yDim);
	GET_INT(NumThreads);
	GET_INT(sizeWeights);
	GET_DBL(dropoutRate);
	GET_INT(dropoutSeed);
	GET_INT(step);

	// Get Data
	GET_DV_VIEW(batchDataStride);
	GET_V_VIEW(xData_t);
	GET_V_VIEW(yData_t);

	// Get input
	GET_V_VIEW(weights);

	// Get derivative
	GET_V_VIEW(dweights_thread);

	// Get loss function
	GET_V_VIEW(loss_thread);

#ifdef USE_OMP
	omp_set_num_threads(NumThreads);
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
	for (int thread_id = 0; thread_id < NumThreads; thread_id++)
	{
		// Get thread local network
		LstmLayer<TYPENAME>* lstm[MAX_LAYER];
		DropoutLayer<TYPENAME>* dropout[MAX_LAYER];
		TYPENAME* ptr_weights = _weights;
		TYPENAME* ptr_dweights = _dweights_thread + sizeWeights*thread_id;
		for (int i = 0; i < nLayer; i++)
		{
			lstm[i] = lstm_thread[i][thread_id];
			dropout[i] = dropout_thread[i][thread_id];
			lstm[i]->assign_weights(&ptr_weights);
			lstm[i]->assign_dweights(&ptr_dweights);
		}
		LinearLayer<TYPENAME>* linear = linear_thread[thread_id];
		linear->assign_weights(&ptr_weights);
		linear->assign_dweights(&ptr_dweights);
		
		SoftmaxCriterion<TYPENAME>* softmax = softmax_thread[thread_id];
		softmax->temperature = 1;

		// Get thread local output
		TYPENAME* loss = _loss_thread + thread_id*periods*batchSizeThread;

		// Copy data
		for (int j = 0; j < batchSizeThread; j++)
		{
			int startingPos = (int)_batchDataStride[thread_id*batchSizeThread + j];
			for (int t = 0; t < periods; t++)
			{
				memcpy(lstm[0]->x_t + t*batchSizeThread*xDim + j*xDim, _xData_t + xDim*startingPos + xDim*t, sizeof(TYPENAME)*xDim);
				memcpy(softmax->y_t + t*batchSizeThread*yDim + j*yDim, _yData_t + xDim*startingPos + yDim*t, sizeof(TYPENAME)*yDim);
			}
		}

		// Initiate weights
		for (int i = 0; i < nLayer ; i++)
		{
			lstm[i]->init_dweights();
			
		}
		linear->init_dweights();

		int status;

		// Train
		// Forward
		// First Layer
		status = lstm[0]->forward_pass_T();
		// Layer to Dropout
		dropout[0]->fetch_from_bottom(lstm[0]->h_t);
		status = dropout[0]->forward_pass_T();
		// Future Layer
		for (int i = 1; i < nLayer; i++)
		{
			lstm[i]->fetch_from_bottom_ptr(dropout[i - 1]->x_t);
			status = lstm[i]->forward_pass_T();
			dropout[i]->fetch_from_bottom(lstm[i]->h_t);
			status = dropout[i]->forward_pass_T();
		}
		// Last Layer to Softmax
		linear->fetch_from_bottom_ptr(dropout[nLayer - 1]->x_t);
		status = linear->forward_pass_T();
		softmax->fetch_from_bottom_ptr(linear->y_t);
		status = softmax->eval_loss_T();

		// Output loss
		memcpy(loss, softmax->loss_t, sizeof(TYPENAME)*batchSizeThread*periods);

		// Backward
		status = softmax->eval_dloss_T();
		linear->fetch_from_top_ptr(softmax->dyhat_t);
		status = linear->back_propagation_T();
		// Softmax to Last Layer
		dropout[nLayer - 1]->fetch_from_top(linear->dx_t);
		status = dropout[nLayer - 1]->back_propagation_T();
		lstm[nLayer - 1]->fetch_from_top_ptr(dropout[nLayer - 1]->dx_t);
		status = lstm[nLayer - 1]->back_propagation_T();
		// Last Layer to First layer
		for (int i = nLayer - 2; i >= 0; i--)
		{
			dropout[i]->fetch_from_top(lstm[i + 1]->dx_t);
			status = dropout[i]->back_propagation_T();
			lstm[i]->fetch_from_top_ptr(dropout[i]->dx_t);
			status = lstm[i]->back_propagation_T();
		}

		// Store information for next training
		for (int i = 0; i < nLayer ; i++)
		{
			// lstm[i]->clear_info();
			lstm[i]->store_info(lstm[i]->h_t, lstm[i]->s_t);
			// lstm[i]->store_info(lstm[i]->h_t + ((periods - 1)*lstm[i]->outputStride), lstm[i]->s_t + ((periods - 1)*lstm[i]->outputStride));
		}
	}
	// Update weights
	// Collapse weights w.r.t. thread
	for (int thread_id = 1; thread_id < NumThreads ; thread_id++)
	{
		TYPENAME* dweights_other_thread = _dweights_thread + thread_id*sizeWeights;
#pragma omp parallel for simd
		for (int j = 0; j < sizeWeights; j++)
		{
			_dweights_thread[j] += dweights_other_thread[j];
		}
	}
	// Call optimizer
	optimizer->update_weights(_weights, _dweights_thread, step);
}
