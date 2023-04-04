# Time series analysis

In this package, we model the underlying random process producing the observed time series. This process is assumed to be a continuous random, but the time series we given in input of this code is a discrete time series. The resulting discrete time series can suffer from distorsions due to the sampling or the observation method used to collect the time series. Nevertheless, we can model the underlying random process using two mathematical objects: the autocovariance function and the power spectral density function. In the end, we are interested in inferring the shape and parameters of these two objects.

In general, we only have access to one realisation of this process: the observed time series. This is why through all of this, we will assume the time series to be ergodic, meaning that the time average equals the ensemble average. 


## Autocovariance and power spectrum


Let $\{X(t)\}$ be a random process stationary up to order 2, meaning the mean and covariance do not change over time.

### Autocovariance function

The covariance function associated to this process is given by Equation {eq}`autocovariance`. Where $\mathrm{E}$ is then ensemble average and $\mathrm{Cov}$ is the covariance operator.

```{math}
:label: autocovariance
R(\tau)=\mathrm{Cov}(X_t,X_{t+\tau})= \mathrm{E}[(X(t)-\mu)(X(t+\tau)-\mu)].
```


The variable of the autocovariance is often referred as the time lag $\tau$. The autocovariance function quantifies the degree of relation between the values of $X$ across time. For example, for a white noise process which is a sequence of indepedant and identically distriubuted values, the autocovariance function is a Dirac distribution.

The autocovariance function has the following properties:

- $R(0)=\mathrm{Var}[X_t]$, the variance of the process.
- $|R(\tau)| \leq R(0)$ for all $\tau$
- The autocovariance function is positive semi-definite, this implies that a covariance matrix is always positive semi-definite.
- If the process is real-valued, then $R(\tau)=R(-\tau)$, the autocovariance function is even.




### Power spectral density

The power spectral density or simply power spectrum is defined with a limit when $T$ tends to infinity in Equation {eq}`powerspectraldensity`.


```{math}
:label: powerspectraldensity
   \mathcal{P}(f) = \lim_{T \to +\infty} \dfrac{{\rm E} \left[ \left|\hat{x}_{T}(f)\right|^2 \right] }{2T}.
```

Where $\hat{x}_{T}(\xi)$ is the truncated Fourier transform of $x(t)$, given by Equation {eq}`fouriertransformTrunc`.

```{math}
:label: fouriertransformTrunc
    \hat{x}_{T}(f) = \int_{-T}^{T} x(t)\, e^{-2i\pi f t } {\rm d}t.
```

The autocovariance function $R(\tau)$ and the power spectral density function $\mathcal{P}(f)$ are Fourier pairs as given by Equation {eq}`psdacvf`.

```{math}
:label: psdacvf
\mathcal{P}(f) = \int_{-\infty}^{+\infty} R(\tau) e^{-2i\pi f \tau} {\rm d }\tau \quad  \quad     R(\tau) = \int_{-\infty}^{+\infty} \mathcal{P}(f) e^{2i\pi f\tau} {\rm d }f. 
```

From the previous equations, we can see that the variance of the process is given by the integral of the power spectral density function over the whole frequency range. 

- $R(0)=\mathrm{Var}[X_t]=\int_{-\infty}^{+\infty} \mathcal{P}(f)  {\rm d }f $, the variance of the process.
- $\mathcal{P}(0)=\int_{-\infty}^{+\infty} {R}(\tau)  {\rm d }\tau=\mu^2$, the square mean of the process.
- $\mathcal{P}(f)$ is always positive, this implies that the covariance matrix is always positive semi-definite.

In the case of white noise, the power spectral density is a constant function.

## References

- P. J. Brockwell and R. A. Davis, *Introduction to Time Series and Forecasting*, Springer, 1996.
- Priestley, M.B. (1981) Spectral Analysis and Time Series. Academic Press Inc, New York. 