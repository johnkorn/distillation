from scipy.io import loadmat
import pickle
import numpy as np
from keras.models import Model

from models import build_cnn

def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    ''' Load data in from .mat file as specified by the paper.
        Arguments:
            mat_file_path: path to the .mat, should be in sample/
        Optional Arguments:
            width: specified width
            height: specified height
            max_: the max number of samples to load
            verbose: enable verbose printing
        Returns:
            A tuple of training and test data, and the mapping for class code to ascii value,
            in the following format:
                - ((training_images, training_labels), (testing_images, testing_labels), mapping)
    '''
    # Local functions
    def reshape(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        img.shape = (width,height)
        img = img.T
        img = list(img)
        img = [item for sublist in img for item in sublist]
        return img

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        return render

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]-1:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('bin/mapping.p', 'wb' ))

    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_]
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_] #shift matlab indicies to start from 0

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_]
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_] #shift matlab indicies to start from 0

    # Reshape training data to be valid
    if verbose == True: _len = len(training_images)
    for i in range(len(training_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        training_images[i] = reshape(training_images[i])
    if verbose == True: print('')

    # Reshape testing data to be valid
    if verbose == True: _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        testing_images[i] = reshape(testing_images[i])
    if verbose == True: print('')

    # Extend the arrays to (None, 28, 28, 1)
    training_images = training_images.reshape(training_images.shape[0], height, width, 1)
    testing_images = testing_images.reshape(testing_images.shape[0], height, width, 1)

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)

    return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)

def load_teacher_model(model_path, training_data):
    model = build_cnn(training_data)
    model.load_weights(model_path)
    # remove softmax
    model.layers.pop()
    model = Model(model.input, model.layers[-1].output)
    # now model outputs logits
    return model

def myGenerator(dataset, batch_size):
    while 1:
        n = len(dataset)
        steps = int(n/batch_size)
        for i in range(steps):
            yield i, dataset[i*batch_size:(i+1)*batch_size]

def save_logits(training_data, model_path, outpaths=('train_logits.npy', 'test_logits.npy'), batch_size=100):
    model = load_teacher_model(model_path, training_data)
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data

    train_logits = []
    train_generator = myGenerator(x_train, batch_size)
    n = len(x_train)
    total_steps = int(n / batch_size)
    print("Processing train data logits in %d steps by batches of size %d" % (total_steps, batch_size))
    for num_batch, x_batch in train_generator:
        print("processing batch %d..." % num_batch)
        batch_logits = model.predict_on_batch(x_batch)

        for i in range(len(batch_logits)):
            #train_logits[num_batch*batch_size+i] = batch_logits[i]
            train_logits.append(batch_logits[i])

        if num_batch >= total_steps-1:
            break

    np.save(outpaths[0], train_logits)
    print('Train logits saved to %s' % outpaths[0])

    test_logits = []
    test_generator = myGenerator(x_test, batch_size)
    n = len(x_test)
    total_steps = int(n / batch_size)
    print("Processing test data logits in %d steps by batches of size %d" % (total_steps, batch_size))
    for num_batch, x_batch in test_generator:
        print("processing batch %d..." % num_batch)

        batch_logits = model.predict_on_batch(x_batch)

        for i in range(len(batch_logits)):
            test_logits.append(batch_logits[i])

        if num_batch >= total_steps-1:
            break


    np.save(outpaths[1], test_logits)
    print('Test logits saved to %s' % outpaths[1])
