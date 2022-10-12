* float(X) < !C!,
* float(X) > !C!,
* float(X) == !C!,
* float(X) == !C!,
* float(X) > !C!,
* float(X) < !C!,
* 0 <= float(X) <= 1,
* isinstance(X, bool),
* isinstance(X, int),
* isinstance(X, float),
* isinstance(X, list),
* isinstance(X, dict),
* isinstance(X, str),
* isinstance(X, tuple),
* isinstance(X, numpy.ndarray),
* type(X).__name__ == 'Tensor',
* type(X).__name__ == 'EagerTensor',  / isinstance(x, torch.sparse_coo_tensor)
* type(X).__name__ == 'RaggedTensor', / isinstance(x, torch.sparse_csr_tensor)
* type(X).__name__ == 'SparseTensor',
* len(X) == !C!,
* len(X) < !C!,
* len(X) > !C!,
* X.dtype == tf.float64,  / X.dtype == torch.float64 
* X.dtype == tf.float32,  / X.dtype == torch.float32
* X.dtype == tf.float16,  / X.dtype == torch.float
* X.dtype == tf.int64,  / X.dtype == torch.double
* X.dtype == tf.int32,  / X.dtype == torch.complex64
* X.dtype == tf.int8,  / X.dtype == torch.cfloat
* X.dtype == tf.int16,  / X.dtype == torch.complex128
* X.dtype == tf.uint16,  / X.dtype == torch.cdouble
* X.dtype == tf.uint8,  / X.dtype == torch.float16
* X.dtype == tf.string,  / X.dtype == torch.half
* X.dtype == tf.bool,  / X.dtype == torch.bfloat16
* X.dtype == tf.complex64,  / X.dtype == torch.uint8
* X.dtype == tf.complex128,  / X.dtype == torch.int8
* X.dtype == tf.qint8,  / X.dtype == torch.int16
* X.dtype == tf.quint8,  / X.dtype == torch.short
* X.dtype == tf.qint16,  / X.dtype == torch.int32
* X.dtype == tf.quint16,  / X.dtype == torch.int
* X.dtype == tf.qint32,  / X.dtype == torch.int64
* X.dtype == tf.bfloat16,  / X.dtype == torch.long
* X.dtype == tf.resource,  / X.dtype == torch.qint8
* X.dtype == tf.half,  / X.dtype == torch.uqint8
* X.dtype == tf.variant,  / X.dtype == torch.qint32
						/ X.dtype == torch.quint4x2
						/ X.dtype == torch.bool
* X.shape.rank == 0,      / torch.linalg.matrix_rank(X) == 0
* X.shape.rank > !C!,      / torch.linalg.matrix_rank(X) > !C!
* X.shape.rank < !C!,      / torch.linalg.matrix_rank(X) < !C!
* X.shape[0] == int(!C!),
* X.shape[1] == int(!C!),
* X.shape[2] == int(!C!),
* X.shape[0] > int(!C!),
* X.shape[1] > int(!C!),
* X.shape[2] > int(!C!),
* X.shape[0] < int(!C!),
* X.shape[1] < int(!C!),
* X.shape[2] < int(!C!),
* tf.experimental.numpy.all(X > !C! ) .numpy(),
* tf.experimental.numpy.all(X < !C! ) .numpy(),
* tf.experimental.numpy.any(X == !C!) .numpy(),
tf.experimental.numpy.all(X == !C!) .numpy()
* isinstance(X, sparse_tensor.SparseTensor),
* isinstance(X, ragged_tensor.RaggedTensor),
* all(i > !C! for i in X),
* all(i == !C! for i in X),
* all(i < !C! for i in X),
* all(i != 0 for i in X),
* all(i is not None for i in X),
* all(i > !C! for i in X.shape),
* all(i == !C! for i in X.shape),
* all(i < !C! for i in X.shape),
* all(type(X) == !T!),
* any(type(X) == !T!),
* all([type(x) == !T! for x in X]),
* all([type(x) == !T! for x in X.values()]),
* any([type(x) == !T! for x in X]),
* any([type(x) == !T! for x in X.values()]),
* all([x.dtype == !T! for x in X]),
* any([x.dtype == !T! for x in X]),
* X.isupper(),
* X[0].isupper(),
* X.islower(),
* X[0].islower(),
* X[0] == !C!,
* X[1] == !C!,
* X[-1] == !C!,
* X[-2] == !C!,
* all([x.dtype == !T! for x in X]),
* any([x.dtype == !T! for x in X]),
* X is None,
* X is not None,