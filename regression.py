from sklearn.linear_model import LinearRegression

class UnivariateRegressionModel: 

    def __init__(self, X, y): 
        self.X = X 
        self.y = y
        self.linear_model = self.fit_model()

    def fit_model(self):
        model = LinearRegression() 
        model.fit(self.X, self.y)

        return model

    def get_beta(self): 
        """Returns the beta from the regression model
        """

        return self.linear_model.coef_[0][0]

    def get_r2(self): 

        return self.linear_model.score(self.X, self.y)

    def get_corr_coef(self): 

        r2 = self.get_r2()
        return r2 ** 0.5

    def predict(self, x): 

        return self.linear_model.predict(x)

    def score(self, X, y): 

        return self.linear_model.score(X, y)

    def summary_message(self): 
        beta = self.get_beta()
        r2 = self.get_r2()
        correlation = self.get_corr_coef()

        summary = """
        The beta for the regression is {beta:.2f}
        The r^2 is {r2:.3f}
        The correlation is {correlation:.2f}
        """.format(beta = beta, r2 = r2, correlation = correlation)

        return summary

    def show_summary(self): 
        print(self.summary_message())

    

    
