#tensorflow #keras 

	im_shape = (X_train.shape[1], 1)
	inputs_cnn = Input(shape = (im_shape), name = "inputs_cnn")
	conv1 = Convolution1D(64, (6), activation = "relu", input_shape = im_shape)(inputs_cnn)

Pentru (e.g.) X_train.shape == (21892, 187)
*input_shape* pentru keras.Convolution1D este de forma (X, 1), unde X este numărul de elemente al eșantionului (i.e. 187 pentru acest caz).