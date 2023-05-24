import numpy as np
import pandas as pd
from scipy import sparse

def proc_col(col):
    """Encodes a pandas column with values between 0 and n-1.
 
    where n = number of unique values
    """
    uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx[x] for x in col]), len(uniq)

def encode_data(df):
    """Encodes rating data with continous user and movie ids using 
    the helpful fast.ai function from above.
    
    Arguments:
      df: a csv file with columns userId, movieId, rating 
    
    Returns:
      df: a dataframe with the encode data
      num_users
      num_movies
      
    """
    df_1 = df.copy()
    cols = ['userId','movieId','rating']
    df_1 = df_1[cols]
    user_indices,user_array,num_users = proc_col(df_1.userId)
    movie_indices,movie_array,num_movies = proc_col(df_1.movieId)
    df_1['userId'] = user_array
    df_1['movieId'] = movie_array
    ### BEGIN SOLUTION
    
    ### END SOLUTION
    return df_1, num_users, num_movies

def encode_new_data(df_val, df_train):
    """ Encodes df_val with the same encoding as df_train.
    Returns:
    df_val: dataframe with the same encoding as df_train
    """
    ### BEGIN SOLUTION
    user_dict = proc_col(df_train.userId)[0]
    movie_dict = proc_col(df_train.movieId)[0]
    df_val = df_val[(df_val.userId.isin(user_dict.keys())) & (df_val.movieId.isin(movie_dict.keys()))]
    df_val.loc[:,'userId'] = [user_dict[x] for x in df_val['userId']]
    df_val.loc[:,'movieId'] = [movie_dict[x] for x in df_val['movieId']]
    ### END SOLUTIproc_col(df_train.userId)[0]ON
    return df_val

def create_embedings(n, K):
    """ Create a numpy random matrix of shape n, K
    
    The random matrix should be initialized with uniform values in (0, 6/K)
    Arguments:
    
    Inputs:
    n: number of items/users
    K: number of factors in the embeding 
    
    Returns:
    emb: numpy array of shape (n, num_factors)
    """
    np.random.seed(3)
    emb = 6*np.random.random((n, K)) / K
    return emb


def df2matrix(df, nrows, ncols, column_name="rating"):
    """ Returns a sparse matrix constructed from a dataframe
    
    This code assumes the df has columns: movieID,userID,rating
    """
    values = df[column_name].values
    ind_movie = df['movieId'].values
    ind_user = df['userId'].values
    return sparse.csc_matrix((values,(ind_user, ind_movie)),shape=(nrows, ncols))

def sparse_multiply(df, emb_user, emb_movie):
    """ This function returns U*V^T element wise multi by R as a sparse matrix.
    
    It avoids creating the dense matrix U*V^T
    """
    
    df["Prediction"] = np.sum(emb_user[df["userId"].values]*emb_movie[df["movieId"].values], axis=1)
    return df2matrix(df, emb_user.shape[0], emb_movie.shape[0], column_name="Prediction")

def cost(df, emb_user, emb_movie):
    """ Computes mean square error
    
    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]
    
    Arguments:
      df: dataframe with all data or a subset of the data
      emb_user: embedings for users
      emb_movie: embedings for movies
      
    Returns:
      error(float): this is the MSE
    """
#   #  import pdb;pdb.set_trace()
#     
   # import pdb;pdb.set_trace()
#     pred_rating = [np.dot(emb_user[i],emb_movie[j]) for i,j in zip(df['userId'],df['movieId'])]
#     total_squared_error = sum([(x-y)**2 for x,y in zip(df['rating'],pred_rating)])
#     return total_squared_error/df.shape[0]
    Y = df2matrix(df,emb_user.shape[0],emb_movie.shape[0])
    user_indices = sparse.find(Y)[0]
    movie_indices = sparse.find(Y)[1]
    ratings = sparse.find(Y)[2]
    pred_rating = (emb_user @ emb_movie.T)[user_indices,movie_indices]
    diff = ratings - pred_rating

    
#     ## BEGIN SOLUTION
    
#     ## END SOLUTION
    return sum(np.square(diff))/df.shape[0]
def finite_difference(df, emb_user, emb_movie, ind_u=None, ind_m=None, k=None):
    """ Computes finite difference on MSE(U, V).
    
    This function is used for testing the gradient function. 
    """
    e = 0.000000001
    c1 = cost(df, emb_user, emb_movie)
    K = emb_user.shape[1]
    x = np.zeros_like(emb_user)
    y = np.zeros_like(emb_movie)
    if ind_u is not None:
        x[ind_u][k] = e
    else:
        y[ind_m][k] = e
    c2 = cost(df, emb_user + x, emb_movie + y)
    return (c2 - c1)/e

def gradient(df, Y, emb_user, emb_movie):
    """ Computes the gradient.
    
    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]   
    Arguments:
      df: dataframe with all data or a subset of the data
      Y: sparse representation of df
      emb_user: embedings for users
      emb_movie: embedings for movies
      
    Returns:
      d_emb_user
      d_emb_movie
    """
    user_indices = sparse.find(Y)[0]
    movie_indices = sparse.find(Y)[1]
    ratings = sparse.find(Y)[2]
    pred_rating = (emb_user @ emb_movie.T)[user_indices,movie_indices]
    diff = ratings - pred_rating
    diff_matrix = sparse.csc_matrix((diff,(user_indices,movie_indices)),\
                                    shape=(emb_user.shape[0], emb_movie.shape[0]))
    grad_user = -(2/df.shape[0])*(diff_matrix @ emb_movie)
    grad_movie = -(2/df.shape[0])*(diff_matrix.T @ emb_user)
    
    return grad_user, grad_movie

# you can use a for loop to iterate through gradient descent
def gradient_descent(df, emb_user, emb_movie, iterations=100, learning_rate=0.01, df_val=None):
    """ Computes gradient descent with momentum (0.9) for a number of iterations.
    
    Prints training cost and validation cost (if df_val is not None) every 50 iterations.
    
    Returns:
    emb_user: the trained user embedding
    emb_movie: the trained movie embedding
    """
    if df_val is not None:
        df_val = encode_new_data(df_val, df_train)
    Y = df2matrix(df, emb_user.shape[0], emb_movie.shape[0])
    momentum_user = np.zeros(emb_user.shape)
    momentum_movie = np.zeros(emb_movie.shape)
    for iteration in range(0,iterations):
        grad_user,grad_movie = gradient(df, Y, emb_user, emb_movie)
        momentum_user = 0.9*momentum_user + 0.1*grad_user
        momentum_movie = 0.9*momentum_movie + 0.1*grad_movie
        emb_user = emb_user - learning_rate*momentum_user
        emb_movie = emb_movie - learning_rate*momentum_movie
        if iteration%50 == 0:
            if df_val is not None:
                print(iteration,cost(df, emb_user, emb_movie),cost(df_val, emb_user, emb_movie))
            else:
                print(iteration,cost(df, emb_user, emb_movie),None)
                
            
    ### BEGIN SOLUTION

    ### END SOLUTION
    return emb_user, emb_movie

