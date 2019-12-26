# caffe-conf-matrix
Python layer for the Caffe [Caffe](https://github.com/BVLC/caffe) deep learning framework to compute the accuracy and the confusion matrix.
This layer will print a confusion matrix of the TEST predictions after the whole TEST images have been processed. It will also work as an accuracy layer, providing Caffe with the predictions accuracy on the TEST set.

The usage is very simple, the layer just has to be used as an accuracy layer in the prototxt file like:
	
	layer {
	  type: 'Python'
	  name: 'py_accuracy'
	  top: 'py_accuracy'
	  bottom: 'ip2'
	  bottom: 'label'
	  python_param {
	    # the module name -- usually the filename -- that needs to be in $PYTHONPATH
	    module: 'python_confmat'
	    # the layer name -- the class name in the module
	    layer: 'PythonConfMat'
	    param_str: '{"test_iter":100}'
	  }
	  include {
	    phase: TEST
	  }
	}

There is a working example in the `examples` folder, which must be copied in `caffe/examples` folder in order for the relative paths to work. The file `python_confmat.py` must be copied in `caffe/examples/mnist` to work for the example, but for your own usage you can place it anywhere as long as the path is included in your `$PYTHONPATH`.

The confusion matrix is printed to console and looks like this:

	Confusion Matrix                                                | Accuracy
	------------------------------------------------------------------------
	3438    166     191     16      45      9       136     0       | 85.93 % 
	191     3306    177     1       69      2       15      0       | 87.90 % 
	88      114     3205    34      431     46      80      3       | 80.10 % 
	30      12      98      3735    78      23      24      0       | 93.38 % 
	11      28      437     29      3196    65      45      11      | 83.62 % 
	3       0       64      7       38      3702    8       0       | 96.86 % 
	59      4       79      42      44      5       3234    1       | 93.25 % 
	2       0       29      3       113     9       6       2639    | 94.22 % 
	Number of test samples: 29676
