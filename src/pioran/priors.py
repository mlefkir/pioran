import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats


class Prior:
    def __init__(self, name, params):
        self.name = name
        self.params = params

class NormalPrior(Prior):
    def __init__(self, name, mean, std):

        params = {'mean': mean, 'std': std}
        super().__init__(name, params)

    def sample(self, key, shape=(1,)):
        return jax.random.normal(key, shape=shape) * self.params['std'] + self.params['mean']

    def logpdf(self, x):
        return jstats.norm.logpdf(x, loc=self.params['mean'], scale=self.params['std'])

    def ppf(self, q):
        return jstats.norm.ppf(q, loc=self.params['mean'], scale=self.params['std'])

class LogNormalPrior(Prior):
    def __init__(self, name, sigma, scale):
        params = {'sigma': sigma, 'scale': scale}
        super().__init__(name, params)
    
    def sample(self, key, shape=(1,)):
        return jax.random.lognormal(key, shape=shape,sigma=self.params['sigma'])* self.params['scale']

    
    # def lognorm_logpdf(x,s,scale=1):
    # return jax.lax.cond(x<0,lambda: -jnp.inf,lambda: jnp.log(scale)-jnp.log(jnp.sqrt(2*jnp.pi)*s*x)- (jnp.log(x)-jnp.log(scale))**2 / (2*s**2))

    def logpdf(self, x):
        # return jax.lax.cond(x<=0,lambda: -jnp.inf,lambda: jnp.log(self.params['mean'])-jnp.log(jnp.sqrt(2*jnp.pi)* self.params['std']*x)- (jnp.log(x)-jnp.log(self.params['mean']))**2 / (2*self.params['std']**2))
        # return jnp.where(x<=0, -jnp.inf, jnp.log(self.params['mean'])-jnp.log(jnp.sqrt(2*jnp.pi)* self.params['std']*x)- (jnp.log(x)-jnp.log(self.params['mean']))**2 / (2*self.params['std']**2))
        return jnp.where(x<=0,-jnp.inf,-jnp.log(jnp.sqrt(2*jnp.pi)*self.params['sigma']*x) -(jnp.log(x)-jnp.log(self.params['scale']))**2 / (2*self.params['sigma']**2))
class UniformPrior(Prior):
    def __init__(self, name, low, high):
        params = {'low': low, 'high': high}
        super().__init__(name, params)

    def sample(self, key, shape=(1,)):
        return jax.random.uniform(key, shape=shape) * (self.params['high'] - self.params['low']) + self.params['low']

    def logpdf(self, x):
        return jstats.uniform.logpdf(x, loc=self.params['low'], scale=self.params['high'] - self.params['low'])

class LogUniformPrior(Prior):
    def __init__(self, name, low, high):
        params = {'low': low, 'high': high}
        super().__init__(name, params)
    
    def sample(self, key, shape=(1,)):
        return jnp.exp(jax.random.uniform(key, shape=shape) * (jnp.log(self.params['high']) - jnp.log(self.params['low'])) + jnp.log(self.params['low']))
    
    def logpdf(self, x):
        return jnp.where((x>=self.params['low'])&(x<=self.params['high']), -jnp.log(x*jnp.log(self.params['high']/self.params['low'])), -jnp.inf)



class PriorCollection:
    def __init__(self, priors):
        self.priors = priors

    def sample(self,key, shape=(1,)):

        samples = []
        keys = jax.random.split(key, len(self.priors))
        for (key,prior) in zip(keys,self.priors):
            samples.append(prior.sample(key,shape))
        return jnp.array(samples).T

    def logprior(self, values):
        logprior_total = 0.0
        for i,prior in enumerate(self.priors):
            logprior_total +=prior.logpdf(values[i])
        return logprior_total