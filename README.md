# caffe-conf-matrix
Pyhton layer for the Caffe [Caffe](https://github.com/BVLC/caffe) deep learning framework to compute the accuracy and the confusion matrix.

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
