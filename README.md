# NEDP

+ The implementation of NEDP

###Requirements

+ Python 3.5.2
+ Tensorflow-gpu == 1.10.0
+ numpy
+ scipy
+ matplotlib
+ networkx

### Basic usage

+ Produce representation

```
python3 main.py
```
`--format`：data format，default is `.mat`
`--input`：the input path
`--output`：the output path
`--mode`：the NEDP available modual, `rnn` or `lstm`
`--node_num`：the number of walk
`--path_length`：the length of walk
`--timesteps`：the timesteps 
`--sequences`：the batch of embedding model
`--batches`：the batch of lapeo
`--representation_size`：the dimension of result
`--gen_epoches`：the embedding model's epoches
`--dis_epoches`：the lapeo's epoches
`--epoches`：total epoches

+ Visualization

```
python3 tsne.py 
```
`--emb`，`--label`和`--result`：the path to embedding, label, and results
`--label_format`：label's name in `.mat`
`--idims``—p`：hyper-paramters in `tsne`



