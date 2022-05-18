# not-to-be-shakespeare
Generating Shakespeare text with RNNs

## The RNN model

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 gru (GRU)                   (None, None, 512)         849408    
                                                                 
 gru_1 (GRU)                 (None, None, 512)         1575936   
                                                                 
 time_distributed (TimeDistr  (None, None, 39)         20007     
 ibuted)                                                         
                                                                 
=================================================================
Total params: 2,445,351
Trainable params: 2,445,351
Non-trainable params: 0
_____________________________________
```
