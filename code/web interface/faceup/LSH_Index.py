import math
from numpy import dot
from numpy.linalg import norm
import pickle


class Image:

  def __init__(self,id,features, label = -2):
    self.id = id
    self.features = features
    self.label = label
  
  def get_id(self):
    return self.id
  
  def get_label(self):
    return self.label
  
  def get_features(self):
    return self.features



class HashFunction:

  def __init__(self, k, d, m, s, w = 4 ):
    self.k = k
    self.m = m
    self.d = d
    self.s = s
    self.w = w

    self.b = []
    self.X = []
    for i in range(k):
      #b has a value between 0 and w
      self.b.append( np.random.uniform(0, w))
    self.X = np.array([np.random.normal(loc=m[i],scale=s[i],size=k) for i in range(d)])
    self.X = np.transpose(self.X)
    
  def hash(self, value, i):
    return int( (np.dot(self.X[i], value) + self.b[i])/self.w )

  def index( self, value, max_h_function = math.inf):
    if( len( value ) != self.d  ):
      return False
    dim = min(self.k, max_h_function)
    return "".join([str(self.hash(value,i)) for i in range(dim)])

class HashTable:

  def __init__(self,hash_size,input_dim,mean,std):
    self.table = dict()
    self.g_function = HashFunction(hash_size,input_dim,mean,std)
  
  def add_value(self, number_h_functions, id, features, label = -1):
    image = Image(id,features, label)
    for num_h in number_h_functions: 
      hash_key = self.g_function.index(features, num_h)
      
      if hash_key not in self.table:
        self.table[ hash_key ] = []
      self.table[ hash_key ].append(image)

  def get_values_of_key(self, features, number_h_function):
    hash_key = self.g_function.index(features,number_h_function)
    if hash_key not in self.table:
      return False
    else:
      return self.table[ hash_key ]



class LSHash:
  def __init__(self,num_g_functions,input_dim,hash_size,mean,std, distance = "euclidean"):

    self.hash_size = hash_size
    self.num_g_functions = num_g_functions
    self.input_dim = input_dim #128
    self.init_hash_tables(hash_size,input_dim,mean,std)
    self.distance = distance
    self.number_h_functions = [5]

  

  def print_info(self):
    print( "LSH Information:")
    print( "Hash size: " + str(self.hash_size ) )
    print( "Num g functions: " + str(self.num_g_functions ) )
    print( "Input Dim: " + str(self.input_dim ) )
    print( "Distance type: " + str(self.distance ) )
 
  #hash_tables will contain the HashTable mapped by each g function
  def init_hash_tables(self,hash_size,input_dim,mean,std):
    self.hash_tables = [HashTable(hash_size,input_dim,mean,std) for _ in range(self.num_g_functions)]
  
  def insert_features(self,id, features, label = -3):
    
    for i in range(self.num_g_functions):
      self.hash_tables[i].add_value(self.number_h_functions, id, features, label)
      

  def eval_distance(self, actual_features, target_feature):
    if(self.distance == "euclidean"):
      return np.linalg.norm(actual_features - target_feature)
    elif( self.distance == "cosine"):
#      return (norm(actual_features)*norm(target_feature))/dot(actual_features,target_feature)
      cos = dot(actual_features,target_feature)/(norm(actual_features)*norm(target_feature))
      return -( cos - 1 )
    else:
      return -1
  
  def insertion_point(self,potential_results,candidate,desired_results):
    index = 0
    if (len(potential_results)==0): 
      return 0
    for result in potential_results:
      if ( candidate[0] == result[0] and candidate[1] == result[1]): 
        return -1
      elif (candidate[2] < result[2]): 
        return index
      else:
        index += 1
    if (index>=desired_results):
      return -1
    else:
      return index



  #We find the ids of the images that share the for each g function, all the time, the same hashed key of the features of the image
  def search(self,features,desired_g_functions,desired_h_functions,desired_results = math.inf ):

    potential_results = []
    dim = min(self.num_g_functions, desired_g_functions)
    element_inserted = 0

    for i in range(dim):
      bucket = self.hash_tables[i].get_values_of_key(features,desired_h_functions)
      if bucket == False:
        continue
      for b in bucket:
        element = [b.get_id(),b.get_label(), self.eval_distance(features, b.get_features())]
        insertion = self.insertion_point(potential_results,element,desired_results)
        if (insertion==-1):
          continue
        else:
          potential_results.insert(insertion,element)
          if (element_inserted >= desired_results):
            potential_results.pop()
          else:
            element_inserted += 1

    return potential_results

  def loadIndex(path):
    with open(path, 'rb') as f:
      index = pickle.load(f)
    return index