
Given a vector of inputs $X^{T}= (X_1 , X_2 , ..., X_p)$, we predict the output $Y$ via the model 

$$\hat{Y} = \hat{\beta}_o + \sum\limits_{j=1}^{P} X_{j}\hat{\beta}_j $$

 

If we are thinking in $\mathbb{R}^2$, the term $\hat{\beta}_o$ is the intercept (where the function intercepts the axis $Y$), but in Machine Learning it is know as the _bias_.


A lot of resources also like to include the constant variable 1 in X, include $\hat{\beta}_o$ in the vector of coefficients $\hat{\beta}$, and then write the linear model in vector form as an inner product:

$$\hat{Y} = X^{T}\hat{\beta}$$

where $X^{T}$ denotes the vector or matrix transpose ( $X$ being a column vector). Generally $\hat(Y)$ can be a $K-vector$, and of course $\beta$ would be a $p x K$ matrix of coefficients. In the _(p+1)-dimensional_ input-output space $(X,\hat{Y})$ represents a hyperplane.

### Matrix Formulation

* $\mathbf{\hat{\beta}_o} = \mathbf{w} = [w_0 w_1 ... w_{n+1}]$ its the row-vector of parameters


* $\mathbf{x} = \begin{bmatrix} x_0 \\ \vdots \\ x_n\end{bmatrix} $ it is the column-vector of samples


$$ \hat{y} = 1 w_0 + x_0 w_1 + ... + x_{n} w_{n+1} = \hat{Y} = 1.\hat{\beta}_o + \sum\limits_{j=1}^{P} X_{j}\hat{\beta}_j $$

*it could be like:

* $\mathbf{x} = \begin{bmatrix} 1 \\ x_0 \\ \vdots \\ x_n\end{bmatrix} $ it is the column-vector of samples (this time with _bias_).



$$ [\hat{y}] = [w_0 \ ... \ w_{n+1}] \begin{bmatrix} 1 \\ x_0 \\ \vdots \\ x_n \end{bmatrix} $$

$$ \hat{Y} = X^{T} \hat{\beta} = \mathbf{w} \mathbf{x} $$

There is some difference of notations, being the $w$ one used more on neural networks literature and the $\beta$ one in statistical literature.


If the constant is included in $X$, then the hyperplane includes the origin and is a subspace; if not, it is an affine set cutting the _Y-axis_ at the point $(0, \hat{\beta}_o)$. 

From now on we assume that the intercept is included in $\hat{\beta}$, continuing the notation found in the book _Elements of Statistical Learning_.

Viewed as a function over the _p-dimensional_ input space, $f(X) = X^{T} \beta$ is linear and its gradient
$f'(X)= \frac{d(f(X))}{dx} = \beta$ is a vector in the input space that points to the steepest uphill direction. 

Later we will see that the _algorithm_ gradient descent "goes" in the contrary direction.

