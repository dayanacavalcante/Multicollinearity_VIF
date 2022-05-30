## Multicollinearity of Predictor Variables

Multicollinearity occurs when a model presents variables that are correlated not only with the response variable but also with each other.

We can quantify the severity of multicollinearity in an ordinary least squares regression analysis by calculating the Variance Inflation Factor (VIF).

- How to interpret?

The square root of the VIF indicates how much larger the standard error increases compared if that variable had 0 correlation with other predictor variables in the model.

A VIF between 5 and 10 indicates high correlation, which can be problematic.

- What to do?

If multicollinearity is an issue in your model, you can remove predictor variables that are highly correlated from the model or use partial least squares regression or principal component analysis to reduce the number of predictors to a smaller set of uncorrelated components.

