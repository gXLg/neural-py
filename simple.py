import numpy as np
import itertools
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pickle

layer_conf = [ 4, 5, 2, 1 ]
layer_func = [ "", "sigmoid", "sigmoid", "sigmoid" ]
k = 0.2

# input data
inputs = list ( itertools.product ( [ 0, 1 ], repeat = layer_conf [ 0 ]))
inputs = np.array ( [ np.array ( i, dtype = np.float128 ) for i in inputs ])

#the task
def what ( i ) :
  return int ( i [ 0 ]) ^ int ( i [ 1 ])

# output data
outputs = np.array ( [ [ what ( i )] for i in inputs ])

def reluDer ( x ) :
  x [ x <= 0 ] = 0
  x [ x > 0 ] = 1
  return x

def leakyReluDer ( x ) :
  x [ x <= 0 ] = 0.01
  x[ x > 0] = 1
  return x

class Net :
  def __init__ ( self ) :
    self.layer = [ np.zeros ( i, dtype = np.float128 ) for i in layer_conf ]
    try :
      with open ( "weights_", "rb" ) as file :
        self.weights = pickle.load ( file )
    except :
      self.weights = [ np.random.randn ( layer_conf [ i ], layer_conf [ i + 1 ])
                       .astype ( np.float128 )
                         for i in range ( len ( layer_conf ) - 1 )]
    ###
    self.error_history = [ ]
    self.epoch_list = [ ]

  def activation ( self, x, func, deriv = False ) :
    #back = ( derivative ) if deriv else ( function )
    if func == "sigmoid" :
      back = ( x * (1 - x)) if deriv else ( 1 / (1 + np.exp(-x)))
    elif func == "tanh" :
      back = ( 1 - x ** 2 ) if deriv else (np.tanh(x))
    elif func == "relu" :
      back = ( reluDer ( x )) if deriv else ( ( abs ( x ) + x ) / 2 )
    elif func == "leaky" :
      back = ( leakyReluDer ( x )) if deriv else ( (( x > 0) * x) + ((x <= 0) * x * 0.01))
    return back

  def forward ( self, input ) :
    self.layer [ 0 ] = np.array ( input )
    for i in range ( len ( layer_conf ) - 1 ) :
      self.layer [ i + 1 ] = self.activation ( np.dot ( self.layer [ i ], self.weights [ i ]),
                                               layer_func [ i + 1 ])

  def backward ( self, output ) :
    errors = [ [ ] for _ in range ( len ( layer_conf ))]
    errors [ -1 ] = output - self.layer [ -1 ]
    self.error = errors [ -1 ] [ 0 ]
    for i in range ( len ( layer_conf ) - 1 ) [ : : -1 ] :
      errors [ i ] = np.dot ( self.weights [ i ], errors [ i + 1 ])

    for i in range ( 1, len ( layer_conf )) [ : : -1 ] :
      delta = k * ( errors [ i ] * self.activation ( self.layer [ i ],
                                                     layer_func [ i ], deriv = True ))
      self.weights [ i - 1 ] += np.multiply.outer ( self.layer [ i - 1 ], delta )

  def train ( self, inputs, outputs, cycles = 2000 ) :
    for cycle in range ( cycles ) :
      err = [ ]
      for i in range ( len ( inputs )) :
        self.forward ( inputs [ i ])
        self.backward ( outputs [ i ])
        err.append ( self.error ** 2 )
      self.error_history.append(np.average(np.abs(err)))
      self.epoch_list.append(cycle)

  def predict ( self, input ) :
    self.forward ( input )
    return 0 if self.layer [ -1 ] < 0.5 else 1

print ( "Start" )
nn = Net ( )
nn.train ( inputs, outputs )

for i in inputs :
  print ( nn.predict ( i ), what ( i ))

with open ( "weights_", "wb" ) as file :
  pickle.dump ( nn.weights, file )


plt.figure ( figsize = ( 15, 5 ))
plt.plot ( nn.epoch_list, nn.error_history )
plt.xlabel ( "Epoch" )
plt.ylabel ( "Error" )
plt.savefig ( "plot.jpg" )



height = max ( layer_conf ) * 1000

im = Image.new ( "RGB",
                 ( len ( layer_conf ) * 2000, height ),
                 "#ccccdd" )
draw = ImageDraw.Draw ( im )

layers = [ ]
for i in range ( len ( layer_conf )) :
  current_layer = [ ]
  for j in range ( layer_conf [ i ]) :
    x = i * 2000 + 1000
    y = ( height - 1000 ) / ( layer_conf [ i ] + 1 ) * ( j + 1 ) + 500
    current_layer.append ( ( x, y ))
  layers.append ( current_layer )

for i in range ( len ( layers ) - 1 ) :
  weights = nn.weights [ i ]
  for j, ( x, y ) in enumerate ( layers [ i ]) :
    nod_weights = weights [ j ]
    total = np.sum ( abs ( nod_weights ))
    for k, ( xx, yy ) in enumerate ( layers [ i + 1 ] ) :
      w = 100 * ( abs ( nod_weights [ k ]) / total )
      draw.line ( ( x, y, xx, yy ), fill = "#9999bb", width = w )

for i in layers :
  for x, y in i :
    r = 200
    middle = ( ( x - r, y - r ), ( x + r, y + r ))
    draw.ellipse ( middle, fill = "#aaaabb", outline = "#9999bb", width = 30 )


im.save ( "gg.jpg" )

# For termux users:
#import os
#os.system ( "termux-open plot.jpg" )

#input ( "Press" )
#os.system ( "termux-open gg.jpg" )
