# Kasreh Ezafeh (Genitive Conjunction /e/) in Persian Language
In this project, we intend to identify the conjunction /e/ which is created due to the existence of genitive phrases in Persian language. Since there is no general rule in Persian language to recognize this conjunction, therefore neural models have been used in this repository. The neural model used here is a combination of BERT and BiLSTM networks, which detects this conjunction with high performance.

<br>

## Persian genitive phrases
According to [this source](https://www.dastur.info/persian-grammar/10-attributes-and-attribution/#bb), in Persian language, Genitive is a kind of phrase syntax that sets two constituents in a hierarchical relationship (normally using the enclitical conjunction /-e/).

The superordinate constituent is called the nucleus, which is determined semantically by the second constituent, the modifier. The modifier is a special form of attribute:

![attribute2_EN.png](images/attribute2_EN.png)

Genitive phrases are normally endocentric, which means that the nucleus (or the modifier) is the reference of the genitive phrase. For example, /bærg-e ʧenɒr/ برگِ چنار is a determining of /bærg/ برگ, and /mɒh-e ordibeheʃt/ ماهِ اردیبهشت a specification of /mɒh/ ماه.

Nevertheless, there are some exocentric qualitative genitive phrases in which neither the nucleus nor the modifier is the reference. This occurence can be noted in the following sentences:

کارگران دستِ خالی از اتاقِ رییس بیرون آمدند.

راه درازست و ما پایِ پیاده.

In these examples, /dæst-e xɒli/ دستِ خالی is neither /dæst/ دست nor /xɒli/ خالی, and /pɒ-je piɒdæ/ پایِ پیاده is neither /pɒ/ پا nor /piɒdæ/ پیاده.


### Classifications of Genitive
The genitive can be classified in Persian as follows:

1. <strong>Qualitative Genitive</strong>: In the qualitative genitive, the modifier describes the nucleus. The qualitative genitive phrase and its nucleus are normally noun phrases, while the modifier is normally an adjectival phrase:<br/><br/>
![Qualitative_EN.png](images/Qualitative_EN.png)
<br/><br/>

2. <strong>Substantial Genitive</strong>: In the substantial genitive, the modifier identifies the material of which the nucleus is made up. In Persian, the substantial genitive is always endocentric. A substantial genitive phrase (like its nucleus and modifier) is a noun phrases:<br/><br/>
![substantial_EN.png](images/substantial_EN.png)
<br/><br/>

3. <strong>Explicative Genitive</strong>: In the explicative genitive, the nucleus represents the class of the modifier. In Persian, the explicative genitive is always endocentric. A explicative genitive phrase (like its nucleus and modifier) is a noun phrase:<br/><br/>
![explicative_EN.png](images/explicative_EN.png)
<br/><br/>

4. <strong>Possessive Genitive</strong>:In the possessive genitive, the nucleus belongs or appertains to the modifier. In Persian, the possessive genitive is always endocentric. The nucleus and modifier of the possessive genitive (like the genitive phrase itself) are normally noun phrases:<br/><br/>
![possessive_EN.png](images/possessive_EN.png)
<br/><br/>
5. <strong>Comitative Genitive</strong>:n the comitative genitive, the modifier identifies a process, which accompanies the nucleus. In Persian, the comitative genitive is always endocentric and used syndetical. The comitative genitive phrase (like its nucleus and modifier) is a noun phrase in the singular:<br/><br/>
![comitative_EN.png](images/comitative_EN.png)
<br/><br/>


### Syntax of Genitive
Genitive phrases are normally syndetical (meaning that they have a conjunction). Nevertheless, some qualitative genitive phrases are used asyndetically (often regarding to phonological reasons): /rezɒ bærɒhæni/ رضا براهنی, /mærd-i irɒni/ مردی ایرانی

genitive phrases cannot contain many constituents. Nevertheless, they can be interlaced:

1. Genitive phrases can be used as modifiers of other genitive phrases:<br/><br/>
![modifiers_of_other_genitive.png](images/modifiers_of_other_genitive.png)
<br/><br/>

2. Qualitative genitive phrases can also become nuclei of other genitive phrases: <br/><br/>
![nuclei_of_other_genitives.png](images/nuclei_of_other_genitives.png)
<br/><br/>


In general, if the nucleus ends in /ɒ/, /u/ or /e/, then the letter ِ /e/ becomes ی /je/:<br/><br/>
![e_to_ye.png](images/e_to_ye.png)
<br/><br/>



## Recognition of conjunction /e/ with BERT-BiLSTM neural network

### Installation
We provide instructions how to install dependencies via  pip.
First, clone the repository locally:

```
git clone https://github.com/HRSadeghi/Kasreh_ezafeh.git
```

Change the current directory to NeuralPersianPoet:
```
cd NeuralPersianPoet
```

Now, prepare environment:
```
pip install -r requirements.txt
```


