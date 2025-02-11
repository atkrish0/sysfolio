def weighcov( cov ):
    '''WEIGHT array (N,1) for Global Min Var Portfolio, given cov.'''
    #  Derived in Cochrane (2005), chp. 5, p.83.
    Viv = matrix.invert_pseudo( cov )
    #                  ^in case covariance matrix is ill-conditioned.
    one = np.ones( (cov.shape[0], 1) )
    top = Viv.dot(one)
    bot = one.T.dot(Viv).dot(one)
    return top / bot


def weighcovdata( dataframe ):
    '''WEIGHT array (N,1) for Global Min Var Portfolio, given data.'''
    V = fecon235.fecon235.covdiflog( dataframe )
    return weighcov(V)