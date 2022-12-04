# StockNet Transformer  

**Alumno: Juan Antonio López Rivera**
**Proyecto final de Redes Neuronales Profundas, PCIC UNAM 2023-1**

###    Profesores:
*    Gibran Fuentes Pineda
*    Berenice Montalvo
*    Emilio Morales

Este repositorio contiene el código de StockNet aumentado con una capa Transformer.
StockNet es una red neuronal para la predicción de movimientos de acciones basado en tweets y precios históricos. En esta implementación se agregan dos opciones para reemplazar el módulo auxiliar de atención temporal: una capa de Multihead Attention o bien una capa Transformer.

***Este código por default usa la capa Transformer, para cambiar cuál de estas dos se ocupa solo hay que modificar la linea 942 de Model.py en la función "assemble_graph" y escoger: "self._build_transformer_temporal_att()" o bien "self._build_multihead_temporal_att()"".***

Por limitaciones de espacio éste repositorio no incluye el dataset de tweets y precios históricos, estos se deben descargar y agregar a la carpeta *data/*, están disponibles [aquí](https://github.com/yumoxu/stocknet-dataset). Adicionalmente se requiere descargar y agregar a la carpeta *res/* embeddings pre-entrenados de [GloVe](https://github.com/stanfordnlp/GloVe).


_________________

### Instrucciones originales de StockNet

Yumo Xu and Shay B. Cohen. 2018. [Stock Movement Prediction from Tweets and Historical Prices](http://aclweb.org/anthology/P18-1183). In Proceedings of the 56st Annual Meeting of the Association for Computational Linguistics. Melbourne, Australia, volume 1. [[bib](https://aclanthology.info/papers/P18-1183/p18-1183.bib)]
> Stock movement prediction is a challenging problem: the market is highly *stochastic*, and we make *temporally-dependent* predictions from *chaotic* data. We treat these three complexities and present a novel deep generative model jointly exploiting text and price signals for this task. Unlike the case with discriminative or topic modeling, our model introduces recurrent, continuous latent variables for a better treatment of stochasticity, and uses neural variational inference to address the intractable posterior inference. We also provide a hybrid objective with  temporal auxiliary to flexibly capture predictive dependencies. We demonstrate the state-of-the-art performance of our proposed model on a new stock movement prediction dataset which we collected.

Contacto del autor original: [yumo.xu@ed.ac.uk](mailto:yumo.xu@ed.ac.uk).

## Dependencies
- Python 2.7.11
- Tensorflow 1.4.0
- Scipy 1.0.0
- NLTK 3.2.5

## Directories
- src: source files;
- res: resource files including,
    - Vocabulary file `vocab.txt`;
    - Pre-trained embeddings of [GloVe](https://github.com/stanfordnlp/GloVe). We used the GloVe obtained from the Twitter corpora which you could download [here](http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip).
- data: datasets consisting of tweets and prices which you could download [here](https://github.com/yumoxu/stocknet-dataset).

## Configurations
Model configurations are listed in `config.yml` where you could set `variant_type` to *hedge, tech, fund* or *discriminative* to get four corresponding model variants, HedgeFundAnalyst, TechincalAnalyst, FundamentalAnalyst or DiscriminativeAnalyst described in the paper.

Additionally, when you set `variant_type=hedge, alpha=0`, you would acquire IndependentAnalyst without any auxiliary effects.

## Running

After configuration, use `sh src/run.sh` in your terminal to start model learning and test the model after the training is completed. If you would like to do them separately, simply comment out `exe.train_and_dev()` or `exe.restore_and_test()` in `Main.py`.
