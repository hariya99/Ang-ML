import numpy as np
class linearRegressor:
    """ A generalized class for linear regression machine learning model.

    """

    def __init__(self, features, targets):
        self.features = features
        self.targets  = targets 
        self.num_of_examples = self.features.shape[0]   #number of rows
        self.num_of_features = self.features.shape[1]  # = num of columns
        #weights has to be a column vector with size equal to number of features
        self.weights = np.zeros(self.num_of_features).T    

    
    def linreg_predictor(self):
        """ The method takes in features(x-values) and weights(magnitude of x-values)
            in the form of matrices and predict the outcome by making use of linear 
            regression equation. The general form of equation is 
            y = w0x0 + w1x1 + w2x2 + .... + wnxn
            matrix form is 
                 ----------------->
            y0 = x00 x01 x02... x0n    | w0     
            y1 = x10 x11 x12... x1n    | w1          
            .  .                       |
            .  .                       |
            ym = xn0 xn1 xn2... xnn    v wn
            i.e y = x.w' (feature multiplied by weights transpose)
        """
        predictor = self.features.dot(self.weights)

        return predictor 
    
    def cost_function(self):
        """ This method calculate the difference between expected outcome and predicted
            outcome. model training is all about reducing this cost which is RMS of 
            difference in values.
        """
        cost_J = (0.5*self.num_of_examples)*(np.sum(linreg_predictor()-self.targets))
        return cost_J

    def train_weights(self):
        

