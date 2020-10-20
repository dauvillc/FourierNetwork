# FourierNetwork
Using neural networks to approach the Fourier series expansion of periodic functions.

A periodic real function $$f$$of period $$T$$ can be written as its Fourier Series:

$$f(x)=\sum_{n=-\infty}^{+\infty}\alpha_n\cos(\frac{2\pi}{T}nx)+\beta_n\sin(\frac{2\pi}{T}nx)$$

Therefore $$f$$ may be approximated with a finite sum of trignometric functions, given the corresponding period and  
Fourier coefficients $$\alpha_n$$ and $$\beta_n$$ for $$n\leq n_{max}$$:

$$f(x)=\sum_{n=-n_{max}}^{n_{max}}\alpha_n\cos(\frac{2\pi}{T}nx)+\beta_n\sin(\frac{2\pi}{T}nx)$$

The goal of this project is to determine the Fourier coefficients of a given couple $$(f, T)$$ using a neural  
network.

Note that if the network manages to approximate $$f$$ on a restricted interval by finding its coefficients, it  
necessarily approximates $$f$$ on $$\mathbb{R}$$:
[![Fourier approx using ML](https://github.com/dauvillc/FourierNetwork/blob/main/network_fourier_approx.PNG "Fourier approx using ML")](https://github.com/dauvillc/FourierNetwork/blob/main/network_fourier_approx.PNG "Fourier approx using ML")
