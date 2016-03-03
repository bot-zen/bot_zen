import time

import numpy as np

import theano
import theano.tensor as T

def build_n(input_var=None):
    pass
    #DIMS = 106
    #SEQ_LENGTH = 25
    #NUM_UNITS = 100
    ## Optimization learning rate
    #LEARNING_RATE = .001
    ## All gradients above this will be clipped
    #GRAD_CLIP = 100

    #l_in = lasagne.layers.InputLayer(shape=(None, SEQ_LENGTH, DIMS), input_var=input_var)
    #l_forward = lasagne.layers.LSTMLayer(
    #    l_in, NUM_UNITS, grad_clipping=GRAD_CLIP,
    #    nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
    #l_backward = lasagne.layers.LSTMLayer(
    #    l_in, NUM_UNITS, grad_clipping=GRAD_CLIP,
    #    nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, backwards=True)
    ## concatenate the two
    #l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])

    ### get only the last output from the sequence
    ##l_slice = L.layers.SliceLayer(l_concat, -1)

    ## output layer is a simple dense connection, with DIMS output unit
    #l_out = lasagne.layers.DenseLayer(l_concat, num_units=DIMS, nonlinearity=lasagne.nonlinearities.sigmoid)
    #return l_out

def build_nn():
    from keras.models import Graph, Sequential

    from keras.layers import Convolution1D, Embedding, MaxPooling1D
    from keras.layers.core import TimeDistributedDense, Dropout, Activation
    from keras.layers.recurrent import LSTM

    activation = 'sigmoid'
    output_dim = 256
    pos_output_dim = 50

    words_maxlen = 8

    char_dims = 106
    embedding_size = 64
    chars_maxlen = 8
    vocab_size = char_dims

    lstm_output_dim = output_dim
    lstm_activation = activation
    lstm_inner_activation = 'hard_sigmoid'

    # Convolution
    filter_length = 3
    nb_filter = 64
    pool_length = 2

    # our model is a tad more complicated that Sequentiell can handle
    gmodel = Graph()

    # Three input layers:
    #  - one for character representations
    #  - two for word2vec representations of the empirist and the wikipedia data
    gmodel.add_input(
        name='chars',
        input_shape=(chars_maxlen,))
    gmodel.add_input(
        name='w2v_emp',
        input_shape=(words_maxlen,w2v_emp.layer1_size,))
    gmodel.add_input(
        name='w2v_big',
        input_shape=(words_maxlen,w2v_big.layer1_size,))


    ### CHARS path
    #
    gmodel.add_node(Embedding(input_dim=char_dims, output_dim=embedding_size,
                            input_length=words_maxlen, mask_zero=True),
                    input='chars', name='chars_embedding')
    #gmodel.add_node(Dropout(0.25),
    #               input='chars_embedding', name='chars_dropout')
    #gmodel.add_node(Convolution1D(input_dim=embedding_size,
    #                              nb_filter=nb_filter,
    #                              filter_length=filter_length,
    #                              border_mode='valid',
    #                              activation='relu',
    #                              subsample_length=pool_length),
    #               input='chars_dropout', name='chars_conv1d')
    #gmodel.add_node(MaxPooling1D(pool_length=pool_length),
    #               input='chars_conv1d', name='chars_maxpool')

    #gmodel.add_node(Flatten(),
    #               input='chars_tdd', name='chars_flat')
    gmodel.add_node(LSTM(lstm_output_dim, return_sequences=True),
                input='chars_embedding', name='chars_lstm')
    gmodel.add_node(Dropout(0.25),
                input='chars_lstm', name='chars_dropout')
    #
    ### CHARS path - end


    ### word2vec paths
    #
    gmodel.add_node(
        LSTM(
            lstm_output_dim, return_sequences=True,
            inner_activation=lstm_inner_activation, activation=activation),
        input='w2v_emp',
        name='w2v_emp_lstm')

    gmodel.add_node(
        LSTM(lstm_output_dim, return_sequences=True,
             inner_activation=lstm_inner_activation, activation=activation),
        input='w2v_big', name='w2v_big_lstm')
    #
    ### word2vec paths - end

    # combine (concat) the outputs and continue...
    gmodel.add_node(
        LSTM(lstm_output_dim, return_sequences=True),
        inputs=['w2v_emp', 'w2v_big', 'chars_dropout'],
        mergemode='concat',
        name='gmodel_lstm')

    gmodel.add_node(
        Dropout(0.5),
        input='gmodel_lstm',
        name='gmodel_dropout')

    gmodel.add_node(
        TimeDistributedDense(input_dim=lstm_output_dim,
                             output_dim=pos_output_dim,
                             activation='softmax'),
        input='gmodel_dropout', name='gmodel_tdd')

    # OUTPUT
    #
    gmodel.add_output(
        name='output',
        input='gmodel_tdd')

    gmodel.compile(
        optimizer='adam',
        loss={'output':'categorical_crossentropy'})

    return gmodel

def use_n():
    pass
    #print("Processing Data...")
    #NUM_EPOCHS = 500
    #X_trainn, y_trainn = load_all_trntstd()
    #X_tr, y_tr = list(), list()
    #X_train, y_train = np.array([]), np.array([])
    #for xid,x in enumerate(X_trainn):
    #    X_tr.append(qone_hot_chars(x))
    #    y_tr.append(qone_hot_chars(y_trainn[xid])[0:len(X_tr[-1])])
    #    if len(X_tr[-1]) > len(y_tr[-1]):
    #        X_tr.pop()
    #        y_tr.pop()
    #    #X_train = np.concatenate((X_train, sliding_window(X_tr[xid],25)),0)
    #    #y_train = np.concatenate((y_train, sliding_window(X_tr[xid],25)),0)

    #X_train = np.vstack(X_train)
    #y_train = np.vstack(y_train)
    #print("...Done.")

    ## Prepare Theano variables for inputs and targets
    #input_var = T.tensor3('inputs')
    #target_var = T.ivector('targets')
    #target_var = T.matrix('targets')  # Normally this is ivector, but binary cross-entropy expects matrix
    #network = build_n(input_var)

    ## Define cost function
    #prediction = lasagne.layers.get_output(network)
    #objective = lasagne.objectives.binary_crossentropy(prediction, target_var)
    #loss = objective.mean()

    #params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.adam(loss, params, learning_rate=0.01)

    ## The network output will have shape (n_batch, 1); let's flatten to get a
    ## 1-dimensional vector of predicted values
    #predicted_values = prediction

    ## Compile a function performing a training step on a mini-batch (by giving
    ## the updates dictionary) and returning the corresponding training loss:
    ##train_fn = theano.function([X, target_var], loss, updates=updates)
    #train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

    ## Finally, launch the training loop.
    #print("Starting training...")
    ## We iterate over epochs:
    #for epoch in range(NUM_EPOCHS):
    #    print("Epoch:"+str(epoch))
    #    # In each epoch, we do a full pass over the training data:
    #    train_err = 0
    #    train_batches = 0
    #    start_time = time.time()
    #    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=False):
    #        inputs, targets = batch
    #        train_err += train_fn(inputs, targets)
    #        train_batches += 1

    #    # And a full pass over the validation data:
    #    val_err = 0
    #    val_acc = 0
    #    val_batches = 0
    #    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=False):
    #        inputs, targets = batch
    #        err, acc = val_fn(inputs, targets)
    #        val_err += err
    #        val_acc += acc
    #        val_batches += 1

    #    # Then we print the results for this epoch:
    #    print("Epoch {} of {} took {:.3f}s".format(
    #            epoch + 1, num_epochs, time.time() - start_time))
    #    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    #    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    #    print("  validation accuracy:\t\t{:.2f} %".format(
    #            val_acc / val_batches * 100))

    ## After training, we compute and print the test error:
    #test_err = 0
    #test_acc = 0
    #test_batches = 0
    #for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    #    inputs, targets = batch
    #    err, acc = val_fn(inputs, targets)
    #    test_err += err
    #    test_acc += acc
    #    test_batches += 1
    #print("Final results:")
    #print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    #print("  test accuracy:\t\t{:.2f} %".format(
    #        test_acc / test_batches * 100))
