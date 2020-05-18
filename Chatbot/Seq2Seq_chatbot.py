import numpy as np
import tensorflow as tf
# for tf version in 1.0.x
import re 
# to clean the text and make it simple
import time

"""Dataset used = https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

and there are some alterations did in this to improve efficiency and model performance than the model which is uploaded through
colab.

happy coding"""



lines=open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations=open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# Data Preprocessing

# To first seperate conversations and its respective ID, creating a Dict and storing them in it.
id2line={}
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]=_line[4]
        
# Extracting the conversation Ids
conversation_ids=[]
for conversation in conversations[:-1]:
    _conversation=conversation.split(' +++$+++ ')[-1]
    # to get the last part in conservation
    _conversation=_conversation[1:-1]
    # to remove the square brackets
    _conversation=_conversation.replace("'","")
    _conversation=_conversation.replace(" ","")
    # to remove quotes and space
    """the entire thing in for loop can be written as 
    _conversation=_conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")"""
    conversation_ids.append(_conversation.split(","))

# Preparing X and y ie questions and answers.
questions=[]
answers=[]
for conversation in conversation_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# Cleaning

def clean_text(text):
    text=text.lower()
    text=re.sub(r"i'm","i am", text)
    text=re.sub(r"he's","he is", text)
    text=re.sub(r"she's","she is", text)
    text=re.sub(r"that's","that is", text)
    text=re.sub(r"we'd","we would", text)
    text=re.sub(r"don't","do not", text)
    text=re.sub(r"you're","you are", text)
    text=re.sub(r"workin'","working", text)
    text=re.sub(r"let's","let us", text)
    text=re.sub(r"i'll","i will", text)
    text=re.sub(r"it's","it is", text)
    text=re.sub(r"i've","i have", text)
    text=re.sub(r"what's","what is", text)
    text=re.sub(r"didn't","did not", text)
    text=re.sub(r"where're","where are", text)
    text=re.sub(r"she'll","she will", text)
    text=re.sub(r"he'll","he will", text)
    text=re.sub(r"c'mon","come on", text)
    text=re.sub(r"\'em"," them", text)
    text=re.sub(r"\'ll"," will", text)
    text=re.sub(r"\'ve"," have", text)
    text=re.sub(r"\'re"," are", text)
    text=re.sub(r"\'d"," would", text)
    text=re.sub(r"won't"," will not", text)
    text=re.sub(r"can't"," can not", text)
    text=re.sub(r"[!@#$%^&*()_+-=,./<>?;:|~`]","", text)
    return text

clean_questions=[]
clean_answers=[]

for question in questions:
    clean_questions.append(clean_text(question))

for answer in answers:
    clean_answers.append(clean_text(answer))

# Filtering out the questions and answers that are too short or too long
short_questions = []
short_answers = []
i = 0
for question in clean_questions:
    if 2 <= len(question.split()) <= 25:
        short_questions.append(question)
        short_answers.append(clean_answers[i])
    i += 1
clean_questions = []
clean_answers = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        clean_answers.append(answer)
        clean_questions.append(short_questions[i])
    i += 1
        
# creating a dictonary that maps words to its number of occurances

word2count={}

for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1

# creating 2 dict that maps the question words and the answer words to a unique integer
threshold=15
questionwords2int={}
word_number=0
for word ,count in word2count.items():
    if count >=threshold:
        questionwords2int[word]=word_number
        word_number+=1
answerwords2int={}
word_number=0
for word ,count in word2count.items():
    if count >=threshold:
        answerwords2int[word]=word_number
        word_number+=1

# Adding the last tokens to these 2 dicts
tokens=['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionwords2int[token]=len(questionwords2int)+1
for token in tokens:
    answerwords2int[token]=len(answerwords2int)+1
    
# Creating a inverse dictonary for the answerwords2int dict
answerint2word={w_i:w for w,w_i in answerwords2int.items()}

# Adding <EOS> token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i]+=' <EOS>'

# Translating all the questions and answers into integers
# and replacing all the words that are filtered out by <OUT>
questions_to_int=[]
for question in clean_questions:
    ints=[]
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int['<OUT>'])
        else:
            ints.append(questionwords2int[word])
    questions_to_int.append(ints)
answers_to_int=[]
for answers in clean_answers:
    ints=[]
    for word in answers.split():
        if word not in questionwords2int:
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append(answerwords2int[word])
    answers_to_int.append(ints)

# Sorting questions and answers by length of question
# this is to speed up the training and optimize the model.
sorted_clean_questions=[]
sorted_clean_answers=[]
for length in range(1,25+1):
    for i in enumerate(questions_to_int):
        if len(i[1])==length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])

# Building of Seq2Seq

# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs=tf.placeholder(tf.int32,[None,None],name='input')
    # as we're dealing for inputs, our input is changed to int which is first argument.
    # second is input dimension as we want, so giving as [None,None] will give 2d matrix
    # last argument is name to the input.
    """ Read the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/placeholder"""
    targets=tf.placeholder(tf.int32,[None,None],name='target')
    lr=tf.placeholder(tf.float32,name='learning_rate')
    # this is to hold the learning rate hyper parameter.
    keep_prob=tf.placeholder(tf.float32,name='Keep_prob')
    # Keep prbability is the parameter that is to control dropout rate.
    return inputs, targets, lr, keep_prob

# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    # We're feeding targets inorder to make them in batches.
    # word2int is a dict thta maps tokens to integers
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    # creating it to concat the sos token in front of every answer replacing the last word/last token id in answers.
    # fill function is used to fill with sos
    # 1st argument is dimension. 2nd argument is word2int dict with <SOS> as key
    """ Read the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/fill"""
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    # strided_slice is a fn used in tf to slice the input tensors
    # 1st argument is tensor to be sliced,
    # 2nd is starting row and column
    # 3rd is ending row and column. Giving -1 to slice the last word
    # 4th is slide dimension to get rid off
    """ Read the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/strided_slice"""
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    # Simple concat function in tf.
    """ Read the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/concat"""
    return preprocessed_targets

# Creating the Encoder RNN
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    # rnn_inputs is the inputs for rnn, ie the output from model_input fun
    # rnn_size is the number of input tensor for this encoder_rnn
    # num_layers is the number of layers
    # Keep_prob is dropout control
    # sequence_lenght is the list of the lenght of each question in the batch 
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    """ Read the documentation here: 
        https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell"""
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    """ Read the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper"""
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    # applying the same lstm dropout functionality to every layer in Encoder part of RNN
    """ Read the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell"""
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    # defining the layers to be used in Forward propagation
                                                                    cell_bw = encoder_cell,
                                                                    # defining the layers to be used in Backward propagation
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    # creates the dynamic version of bidirectional RNN network.
    """ Read the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn"""
    return encoder_state


# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    # encoder_state is required to start decoding part
    # decoder_embedded_input is for embedding the descrete objects such as words to vectors of real numbers. Refer tf documentation
    # decoding_scope used to wrap the tf variables
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    # as attention mechanism depends on number of observation == batch size here.
    """ For guidance in Embedding and Variable_scope, read the below documentations,
    https://www.tensorflow.org/programmers_guide/embedding
    https://www.tensorflow.org/api_docs/python/tf/variable_scope"""
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    # attention_keys are the keys to be compared to the target state
    # attention_values are used to construct context vectors from encoder which is used in decoder
    # atention_score_function used to construct similarities between keys and the target state
    # attention_construct_function is used to build the attention state.
    """ Read the documentation here:
        https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/prepare_attention"""
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    """ Read the documentation here:
        https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/attention_decoder_fn_train"""
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    """ Read the documentation here:
        https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder"""
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    # Simple dropout
    """ Read the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/nn/dropout"""
    return output_function(decoder_output_dropout)
 
# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    # num_word isthe max length of answer
    """ Found a good article for model structuring:
        http://web.stanford.edu/class/cs20si/lectures/notes_04.pdf"""
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    """ Find the documentation here:
        https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/attention_decoder_fn_inference"""
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
 
# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        """ Find the documentation here:
            https://www.tensorflow.org/api_docs/python/tf/truncated_normal_initializer"""
        biases = tf.zeros_initializer()
        """ Find the documentation here:
            https://www.tensorflow.org/api_docs/python/tf/zeros_initializer"""
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        # normalizer is set as none here.
        """ Find the documentation here:
            https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected"""
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
 
# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    # Answers_num_words is the total number of words in answers
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                         answers_num_words + 1,
                                                         encoder_embedding_size,
                                                         initializer = tf.random_uniform_initializer(0, 1))
    """ Find the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence"""
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    """ Find the documentations here:
        https://www.tensorflow.org/api_docs/python/tf/Variable
        https://www.tensorflow.org/api_docs/python/tf/random_uniform"""
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    """ Find the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup"""
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions

# Training the Seq2Seq model

# Setting the Hyperparameters
epochs = 100
batch_size = 32
rnn_size = 1024
num_layers = 3
encoding_embedding_size = 1024
decoding_embedding_size = 1024
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5
# As per Jefree Hinton (deep Learning Guru) keeping Keep_probability(dropout for Hidden layer) to 50%
""" His paper: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf"""
 
# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()
""" Find the documentations here:
    https://www.tensorflow.org/api_docs/python/tf/reset_default_graph
    https://www.tensorflow.org/api_docs/python/tf/InteractiveSession"""

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()
 
# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')
""" Find he documentation here:
    https://www.tensorflow.org/versions/r0.12/api_docs/python/io_ops/placeholders#placeholder_with_default"""   

# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)
""" Find the documentation here:
    https://www.tensorflow.org/api_docs/python/tf/shape"""
 
# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       # Reshaping of the input
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerwords2int),
                                                       len(questionwords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionwords2int)
""" Find the documentations here:
    https://www.tensorflow.org/api_docs/python/tf/reverse 
    https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.reshape.html"""

# Setting up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    """ Find the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/name_scope"""
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    """ Find the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss
        https://www.tensorflow.org/api_docs/python/tf/ones"""
    optimizer = tf.train.AdamOptimizer(learning_rate)
    """ Find the documentation here:
        https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer"""
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    """ Find the documentations here:
        https://www.tensorflow.org/versions/r0.12/api_docs/python/train/gradient_clipping
        https://www.tensorflow.org/api_docs/python/tf/clip_by_value"""
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

# Padding the sequences with the <PAD> token
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
# this is to maintain length of the question sequence and answer sequence as same.
# Q: [who, are, you, <EOD>, <PAD>, <PAD>]
# A: [I, am, vignesh, from, trichy, <EOD>]
    
# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionwords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerwords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
# return sends a specified value back to its caller, whereas yield can produce a sequence of values. so using yield here
""" Read this for perfect definition:
    http://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/"""
  
# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15) # 15% here
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]
 
# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "./chatbot_weights.ckpt" # For Windows users, replace this line of code by: checkpoint = "./chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
""" Find the documentation here:
    https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer"""
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")

""" Find the documentation for advanced print statements:
    https://pyformat.info/"""

# Testing the chatbot 
 
# Loading the weights and Running the session
checkpoint = r"checkpoint.ckpt"
session = tf.InteractiveSession()
""" Find the documentation here:
    https://www.tensorflow.org/api_docs/python/tf/InteractiveSession"""
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
""" Find the documentation here:
    https://www.tensorflow.org/api_docs/python/tf/train/Saver"""
saver.restore(session, checkpoint)
 
# Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
 
# Setting up the chat
while(True):
    question = input("You: ")
    if question.lower() == 'bye':
        break
    question = convert_string2int(question, questionwords2int)
    question = question + [questionwords2int['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answerint2word[i] == 'i':
            token = ' I'
        elif answerint2word[i] == '<EOS>':
            token = '.'
        elif answerint2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answerint2word[i]
        answer += token
        if token == '.':
            break
    print('Bot: ' + answer)
