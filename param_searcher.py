import random, json
import numpy as np
from math import floor

''' This script generates random hyperparameters from options, for hyperparameter search.
    Info:
        - 50/50 sgd or adam
        - 20/80 vanilla gradient / normalized gradient
        - learning rate: chosen step-logarithmically between 1e-2 and 1e-4
        - network type: one to three layers, convolutional or multilayer perceptrons.
        - layer sizes: chosen between 40 and 640

        - init type: initialize weight values as random uniform or truncated gaussian
        - weight scales: scale all weights by a factor between 31 and 0.00031.
        - init var(iance): use xavier (1/n_in) or sqrt (1/sqrt(n_in)) initialization variance

'''
    

def get_params():
    args = {}
    args['opt'] = 'sgd' if random.random()<0.5 else 'adam'
    args['lr'] = random.choice([0.1**i for i in np.linspace(2,4,9)])
    args['net type'] = np.random.choice([['mlp','mlp'], ['mlp','mlp','mlp']])
#        ['conv','mlp'], ['conv','mlp','mlp'], ['conv','conv','mlp']])
#args['kernel size'] = np.random.choice([1,2,4,6,8,12,16], 3)

    num_layers = len(args['net type'])

    args['layer sizes'] = []
    for i in range(num_layers-1):
        args['layer sizes'].append( int(floor(2**random.choice([5,6,7,8,8.5,9,9.5])\
                + 2**random.randrange(3,8) )))

    args['init type'] = np.random.choice(['unif', 'trunc normal'], num_layers).tolist()
    args['weights scales'] = np.random.choice([0.1**i for i in \
                             np.arange(-1.5,3.5,0.5)], num_layers).tolist()
    args['init var'] = np.random.choice(['xav in', 'sqrt in', 'glorot'], num_layers).tolist()

    args['base normalization'] = np.random.choice(['meansum']).tolist()
    X = True
    while X:
        args['normalization beta: powers'] = np.random.choice(\
                ['log','1','abs','2','0.5','max','logsumexp'], 3).tolist()
        for a in args['normalization beta: powers']:
            if X and not a=='1': X = False

    args['normalization alpha: scaling'] = np.random.choice([\
            np.random.choice(['1','1e-1','1e-2','1e-3','1e-4',\
            '3e-6','1e1','3e2']),
            '1 / '+str(np.random.choice([0.01,0.1,1,10,100,1000])) +\
            ' ' + np.random.choice(['grad size', 'grad size^2', \
            'sqrt grad size', 'log grad size'])], p=[0.3,0.7])

    # This normalization keeps the norm of the gradient at least a value c.
    args['normalization gamma: threshold'] = np.random.choice(\
                [0.1**i for i in np.arange(-1, 5, 0.5)])


    print("Parameters:")
    print(json.dumps(args, sort_keys=True, indent=4, separators=(',',': ')))

    return args

if __name__=='__main__':
    get_params()
