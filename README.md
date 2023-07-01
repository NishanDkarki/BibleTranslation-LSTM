# BibleTranslation-LSTM
This is a Machine Learning project with the objective to generate language translation across multiple languages.

Running the model
The model was created in the Jupyter Notebook.
To run the model:
1. Copy the attached folder to the local PC: BibleTranslation-LSTM
2. Open Jupyter Notebook
3. Open the folder Project|Code.
4. Open DMFinalProjectTeamBearGrills.ipynb in Jupyter Notebook

To load the saved model
1. Go to the ”Loading a saved model” section at the end of the .ipynb file
2. Run ”reloadedtranslator” to load the translator folder
3. All files are included in the folder.


1 ABSTRACT
Neural Machine Translation has varieties of applications in various domains and there has been a lot of
research performed on this field. In the paper, we propose a multi-lingual model that will be able to
translate from any source language to any target language, even though the model has not been trained on
the language pair before (zero-shot translation). For the model, we used a simple RNN architecture and
implemented a global attention layer. For training, we used Opus-100 datasets for the model.
For simplicity, we trained our model with five pair of language sets and evaluated the results.

2 Introduction
NMT (Neural Machine Translation) has been widely used for language translation and has produced some
state of art results. The encoder/decoder model of NMT makes it a good candidate for translation and
is simple to implement. With the introduction of the attention mechanism, the effectiveness of the NMT
model has improved extensively, which encouraged us to deploy the model architecture. The SICEM Neural
The machine Translation project implements an RNN model with an additive attention layer. The model is
trained using the language from the datasets in multiple languages, to translate to desired language. The
input of the model will be a token of the target language and an input sentence, which is a sequence of
words. The output will be the translation of the sentence in the desired language as shown below.

3 Related Work
The NLP deploys different implementations of algorithms to draw out meaningful information from the
text source. These algorithms are implemented through machine learning models. RNN and CNN have
been a state of art implementations for NLP tasks. Though RNN has been an easy choice for NLP problems,
it has difficulties while predicting longer texts. To alleviate the shortcomings of RNN, LSTM has been
introduced. Though, LSTM mitigated the problem with RNN, the initial issues of long-term dependencies
and parallelization persisted. The introduction of the attention mechanism minimized the problem with long-term
dependencies. Different global and local attention mechanism has since been implemented
to generate some improved results in language translation.

4 Dataset Collection
For our training data, we used Opus-100 dataset which is being downloaded from opus website
(https://opus.nlpl.eu). The Opus represents more than 100 languages and more than fifty-five millions 
pair of sentences. For our training purposes, we used the following language pairs.
• English to Portuguese
• English to Spanish
• English to French
• English to Turkish
• English to Italian

We have taken 75,000 sentences from each of the languages. We combined all the files into a single file
containing all five language pairs and 375,000 sentences altogether. This file has been named as
“source.txt”:
• < 2es > I am going to school
• < 2pt > No one is at home
• < 2it > This is my life
• < 2tr > I like food
Similarly, “target.txt” file has been created which will contain the translation of the source languages.

5 Data pre-processing
For the data preprocessing, we imported the data from text files into a string array. We converted the
dataset into a numpy array and the TensorFlow text functions were used to pre-process the dataset. The
dataset has been converted into lowercase for uniformity. We have also removed whitespace, special
characters and duplicate sentences. We added ”start” and ”end” tokens to each of the sentences.

After the data was standardized, we built a vocabulary for the input and target sentences. We used the
textvectorization.getvocabulary() function from TensorFlow text to build the vocabulary. The vocabulary
consists of all the unique words of the input and target datasets. We set our vocab size to 20,000. This
number was generated using the textvectorization.getvocabularysize() from tensorflow text. We estimated
each language to have around 4,000 unique vocab words.

For the vectorization, we used tensorflow.textvectorization to generate token IDs’ for each sentence.
The output vector after tokenization has a shape equals to the batch size and length of the longest sentence
in the input dataset. For instance: vector shape = (length of the longest sentence, batch size = 128 ).
Every sentence in the dataset was padded with 0 to maintain the vector shape.

6 Model Creation
For the model selection, we researched different sequence-to-sequence models. We decided to use an RNN
model with an encoder ad decoder with a global attention layer, as inspired by the paper ”Effective
Approaches to Attention-based Neural Machine Translation”. The model consists of three parts.
• Encoder layer
• Attention layer
• Decoder layer
We used an RNN architecture.

6.1 Encoder Layer
We have two layers in our encoder
• Embedding layer
• GRU layer
The token that has been fed into the encoder, and embedding layer creates an embedding vector using the
token. Then the GRU layer outputs the state and the sequence using the output space dimensionality.
This is a kind of RNN layer that has the activation function ”Tanh” and only holds the memory of the
previous state. It trains faster and gives better results than LSTM and therefore, we preferred to use this
implementation.

6.2 Attention Layer
The decoder uses the attention weights to focus on parts of the input sentence. 
The attention layer predicts one word at a time for the word it is being fed by processing all the hidden
state and the encoder output.

6.3 Decoder Layer
This layer is responsible for predicting the output tokens. The decoder runs in a loop in this model rather
than calling it once. The encoder output is fed into the decoder. RNN keeps track of what has been
generated thus far. RNN output is used by the decoder to produce the context vector. Then it combines
the encoder output and the context vector to the product attention vector which is then used to predict the next
token.

7 Train the Model
For training our model, we trained our model with 375,000 pairs of languages where five languages are
represented with 75,000 pairs each. We trained our model with a batch size of 128 and the model was trained
for 3 epochs. We used Categorical Cross entropy as our loss function as our data is sparse. For, the
optimizer, we have used Adam. And the batch size is 128. Here is the summary of the training of the model.
• Take inputs
• Vectorize the text and convert it into a mask
• Feed them into the encoder in batch
• Loop through each of the target tokens
– Decoder will run one step at a time
– SparseCategoricalCrossentropy loss will be measured at each of the steps
– Calculate the average loss
• Using the Adam optimizer, we try to minimize the overall loss.
To run for 3 epochs it took almost 17.8 hours to get trained.

The final batch loss is 2.8135. Due to hardware limitations, we could not run it for more epochs. We could
have further minimized the loss by running the model for greater epochs. Once the model is finished training,
we saved it for further use, so that we no longer need to train it again unless new data is required to be fed.

8 Experiments and Results
After training the model, we can do some experiments with some inputs. From the experiment, we can see
that for the small sentences, the model worked reasonably well but for the long sentences the model’s focus
is distributed and the prediction is not that good.

9 Conclusion
In conclusion, we presented a model which has been trained with five datasets. We can feed more datasets
into the model for improved translations. Moreover, an increased number of epochs could bring better results.
For the intensive learning of our model, powerful hardware could be used so that the training time could be
reduced when the model will be trained with lots of data. Furthermore, we can also improve the model by
incorporating self-attention in the encoder and decoder. Currently, the model has access to its previous
RNN state output. Finally, we want to build a website where users will be able to read the Bible in any
language as per the user’s preferences and the translation will be generated with our SICEM Neural
Machine Translation.

References
[1] Aharoni, Roee, Melvin Johnson, and Orhan Firat. ”Massively multilingual neural machine translation.”
arXiv preprintarXiv:1903.00089 (2019).
[2] Johnson, Melvin, et al. ”Google’s multilingual neural machine translation system: Enabling zero-shot
translation.” Trans-actions of the Association for Computational Linguistics 5 (2017): 339-351.
[3] arXiv:1706.03762 [cs.CL]
[4] Minh-Thang Luong, Hieu Pham, Christopher D. Manning. ”Effective Approaches to Attention-based
Neural Machine Translation”
10
