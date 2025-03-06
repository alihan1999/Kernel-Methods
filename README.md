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

[![\\ \mathcal{H}_{S}=\{f\in \mathcal{H} | f=\sum_{i=1}^{n}a_iK_{X_i}\}](https://latex.codecogs.com/svg.latex?%5C%5C%20%5Cmathcal%7BH%7D_%7BS%7D%3D%5C%7Bf%5Cin%20%5Cmathcal%7BH%7D%20%7C%20f%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7Da_iK_%7BX_i%7D%5C%7D)](#_)

Note that [![\\ f^\perp(x_i)=\langle f^\perp,K_{x_i} \rangle_{\mathcal{H}} = 0](https://latex.codecogs.com/svg.latex?%5C%5C%20f%5E%5Cperp(x_i)%3D%5Clangle%20f%5E%5Cperp%2CK_%7Bx_i%7D%20%5Crangle_%7B%5Cmathcal%7BH%7D%7D%20%3D%200)](#_)
