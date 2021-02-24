# pyBTM
Python wrapper for Biterm Model algorithm (Yan et al., 2013)  
BTM learning and inference algorithm source code can be found here: https://github.com/xiaohuiyan/BTM  
&nbsp;

### Demo
See demo code in demo/ for more usage information

### Data
Each corpus is expected to be in a single plain text file
Words should be separated by spaces ' '
Documents should be separated by new lines '\n'

### Initialization
Initialize a new model:
```
btm = pyBTM.BTM(K=k, input_path=path_to_data, alpha=alpha, beta=beta, niter=niter)
```

Encode data into bag-of-words format
```
btm.index_documents()
```
### Learning
Learn topics:
```
btm.learn_topics()
```

### Inference
Infer topics for each document:
```
btm.infer_documents()
```

### Evaluate Performance
Evaluate performance of the model using CV and UMass coherence scores:
```
coherence_models = btm.build_coherence_model(measures=['c_v', 'u_mass'])
print([*map(lambda e : (e[0], e[1][1]), cm_dict.items())])
```

### View Results
Show top 10 words for each topic:
```
topics = btm.get_topics(include_likelihood=False, use_words=True, L=10)
```

Extra model information
```
print(btm.info())
```

#### References
Yan, X., Guo, J., Lan, Y., & Cheng, X. (2013). A Biterm Topic Model for Short Texts.
