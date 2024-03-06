
priors
======

.. py:module:: pioran.priors


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`Prior <pioran.priors.Prior>`
     - \-
   * - :py:obj:`NormalPrior <pioran.priors.NormalPrior>`
     - \-
   * - :py:obj:`LogNormalPrior <pioran.priors.LogNormalPrior>`
     - \-
   * - :py:obj:`UniformPrior <pioran.priors.UniformPrior>`
     - \-
   * - :py:obj:`LogUniformPrior <pioran.priors.LogUniformPrior>`
     - \-
   * - :py:obj:`PriorCollection <pioran.priors.PriorCollection>`
     - \-




Classes
-------

.. py:class:: Prior(name, params)




.. py:class:: NormalPrior(name, mean, std)

   Bases: :py:obj:`Prior`


   .. rubric:: Overview


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`sample <pioran.priors.NormalPrior.sample>`\ (key, shape)
        - \-
      * - :py:obj:`logpdf <pioran.priors.NormalPrior.logpdf>`\ (x)
        - \-
      * - :py:obj:`ppf <pioran.priors.NormalPrior.ppf>`\ (q)
        - \-


   .. rubric:: Members

   .. py:method:: sample(key, shape=(1, ))


   .. py:method:: logpdf(x)


   .. py:method:: ppf(q)




.. py:class:: LogNormalPrior(name, sigma, scale)

   Bases: :py:obj:`Prior`


   .. rubric:: Overview


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`sample <pioran.priors.LogNormalPrior.sample>`\ (key, shape)
        - \-
      * - :py:obj:`logpdf <pioran.priors.LogNormalPrior.logpdf>`\ (x)
        - \-


   .. rubric:: Members

   .. py:method:: sample(key, shape=(1, ))


   .. py:method:: logpdf(x)




.. py:class:: UniformPrior(name, low, high)

   Bases: :py:obj:`Prior`


   .. rubric:: Overview


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`sample <pioran.priors.UniformPrior.sample>`\ (key, shape)
        - \-
      * - :py:obj:`logpdf <pioran.priors.UniformPrior.logpdf>`\ (x)
        - \-


   .. rubric:: Members

   .. py:method:: sample(key, shape=(1, ))


   .. py:method:: logpdf(x)




.. py:class:: LogUniformPrior(name, low, high)

   Bases: :py:obj:`Prior`


   .. rubric:: Overview


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`sample <pioran.priors.LogUniformPrior.sample>`\ (key, shape)
        - \-
      * - :py:obj:`logpdf <pioran.priors.LogUniformPrior.logpdf>`\ (x)
        - \-


   .. rubric:: Members

   .. py:method:: sample(key, shape=(1, ))


   .. py:method:: logpdf(x)




.. py:class:: PriorCollection(priors)


   .. rubric:: Overview


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`sample <pioran.priors.PriorCollection.sample>`\ (key, shape)
        - \-
      * - :py:obj:`logprior <pioran.priors.PriorCollection.logprior>`\ (values)
        - \-


   .. rubric:: Members

   .. py:method:: sample(key, shape=(1, ))


   .. py:method:: logprior(values)







