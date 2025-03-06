# Kernel-Methods
Special thanks to [@ostadgeorge](https://github.com/ostadgeorge) for the motivation :)

# Kernel Ridge Playground
A notebook to experiment ridge regression with various kernels and lambdas

It also includes the implementation of kernel ridge from scratch

Here's a sample output:

![sample](./images/kernel_ridge.png)


## What are we doing?!
We are optimizing:

[![\\ 	\arg\min\limits_{f\in \mathcal{H}}\frac{1}{n}\sum_{i=1}^{n}(y_i-f(x_i))^2+\lambda \|f\|^2_{\mathcal{H}}](https://latex.codecogs.com/svg.latex?%5C%5C%20%09%5Carg%5Cmin%5Climits_%7Bf%5Cin%20%5Cmathcal%7BH%7D%7D%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D(y_i-f(x_i))%5E2%2B%5Clambda%20%5C%7Cf%5C%7C%5E2_%7B%5Cmathcal%7BH%7D%7D)](#_)

We can write the direct sum [![\\ \mathcal{H}=\mathcal{H}_{S} \oplus \mathcal{H}^\perp_{S}](https://latex.codecogs.com/svg.latex?%5C%5C%20%5Cmathcal%7BH%7D%3D%5Cmathcal%7BH%7D_%7BS%7D%20%5Coplus%20%5Cmathcal%7BH%7D%5E%5Cperp_%7BS%7D)](#_)
where 

[![\\ \mathcal{H}_{S}=\{f\in \mathcal{H} | f=\sum_{i=1}^{n}a_iK_{X_i}, a_i\in\mathbb{R}\}](https://latex.codecogs.com/svg.latex?%5C%5C%20%5Cmathcal%7BH%7D_%7BS%7D%3D%5C%7Bf%5Cin%20%5Cmathcal%7BH%7D%20%7C%20f%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7Da_iK_%7BX_i%7D%2C%20a_i%5Cin%5Cmathbb%7BR%7D%5C%7D)](#_)

Note that [![\\ f_{\mathcal{H}^\perp_s}(x_i)=\langle f_{\mathcal{H}^\perp_s},K_{x_i}\rangle=0](https://latex.codecogs.com/svg.latex?%5C%5C%20f_%7B%5Cmathcal%7BH%7D%5E%5Cperp_s%7D(x_i)%3D%5Clangle%20f_%7B%5Cmathcal%7BH%7D%5E%5Cperp_s%7D%2CK_%7Bx_i%7D%5Crangle%3D0)](#_)

so 

[![\\ f(x_i)=f_{\mathcal{H}_s}(x_i)=\sum_{j=1}^{n}a_jK(x_i,x_j)](https://latex.codecogs.com/svg.latex?%5C%5C%20f(x_i)%3Df_%7B%5Cmathcal%7BH%7D_s%7D(x_i)%3D%5Csum_%7Bj%3D1%7D%5E%7Bn%7Da_jK(x_i%2Cx_j))](#_)

Using matrix notation, we can write the first term of our objective as 

[![\\ \frac{1}{n}(Ka-Y)^\top(ka-Y)](https://latex.codecogs.com/svg.latex?%5C%5C%20%5Cfrac%7B1%7D%7Bn%7D(Ka-Y)%5E%5Ctop(ka-Y))](#_)

Moreover, note that 

[![\\ \|f\|^2_{\mathcal{H}}=\langle f,f\rangle_\mathcal{H} = \langle f_{\mathcal{H}_s}+ f_{\mathcal{H}^\perp_s},f_{\mathcal{H}_s}+ f_{\mathcal{H}^\perp_s}\rangle_\mathcal{H}=\|f_{\mathcal{H}_s}\|^2_{\mathcal{H}} + \|f_{\mathcal{H}^\perp_s}\|^2_{\mathcal{H}}](https://latex.codecogs.com/svg.latex?%5C%5C%20%5C%7Cf%5C%7C%5E2_%7B%5Cmathcal%7BH%7D%7D%3D%5Clangle%20f%2Cf%5Crangle_%5Cmathcal%7BH%7D%20%3D%20%5Clangle%20f_%7B%5Cmathcal%7BH%7D_s%7D%2B%20f_%7B%5Cmathcal%7BH%7D%5E%5Cperp_s%7D%2Cf_%7B%5Cmathcal%7BH%7D_s%7D%2B%20f_%7B%5Cmathcal%7BH%7D%5E%5Cperp_s%7D%5Crangle_%5Cmathcal%7BH%7D%3D%5C%7Cf_%7B%5Cmathcal%7BH%7D_s%7D%5C%7C%5E2_%7B%5Cmathcal%7BH%7D%7D%20%2B%20%5C%7Cf_%7B%5Cmathcal%7BH%7D%5E%5Cperp_s%7D%5C%7C%5E2_%7B%5Cmathcal%7BH%7D%7D)](#_)

When searching in $\mathcal{H}_s$, we know that the second term is equal to zero, and the optimal $f$ lies in the subspace $\mathcal{H}_s$, which means that we can now work with a finite dimensional space aka $\mathcal{H}_s$.


Hence, we reformulate the original problem as

[![\\ \arg\min\limits_{a\in\mathbb{R}^n}\frac{1}{n}(Ka-Y)^\top(Ka-Y)+\lambda a^\top Ka](https://latex.codecogs.com/svg.latex?%5C%5C%20%5Carg%5Cmin%5Climits_%7Ba%5Cin%5Cmathbb%7BR%7D%5En%7D%5Cfrac%7B1%7D%7Bn%7D(Ka-Y)%5E%5Ctop(Ka-Y)%2B%5Clambda%20a%5E%5Ctop%20Ka)](#_)



Using convexity and basic matrix calculus, we can find the optimal $a$ as

[![\\ a=(K+\lambda n I)^{-1}Y](https://latex.codecogs.com/svg.latex?%5C%5C%20a%3D(K%2B%5Clambda%20n%20I)%5E%7B-1%7DY)](#_)


## Benefits
- Theoretical: With this formulation, we make our model more robust, since lower norm makes the model less sensitive to small perturbations
- Practical: Searching in a finite dimensional space makes it possible for the computer to represent the answer in terms of a finite number of coefficients.
