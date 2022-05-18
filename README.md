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

## Generated Text

Even though mostly the words don't make sense, it is interesting that model can generate some words and meaningful sentences from character level training data.

```
rticus: i, without note,-here's
a vertel tears with smiljnech.

second officer:
faith, there had been many, or elumy to reward
whihe he remember'd.
a very on your actions and daugk,
that may fully tubly care edsured here's anly arm detter and the bleared sightry
sevond the common people.

second officer:
has he did budgen deeds doull

brutus:
i will give them make i as liqy as little question
as he is proud to do't.

brutus:
what's the mad me clip than a never o hate
he will not bloody bleading:
if he did so did at the common disposition.

sicinius:
he cannot temperately that may fully discover his
the arm our stand, as bard as he hath
displeasure your sulvessers: set him speak: matrons flung gloves,
let country? he was he wounded?
god sand carry with us;
for sinking under thee; you are knowen part of your ay, such a nettle but they
plasing beee: they love or hate
him men true.
where is he wounded?
god save you give me to care whether
the people is tho market-place nor on him our
putter
```
