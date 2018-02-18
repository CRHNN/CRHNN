import tensorflow as tf
import os
import doc2vec
class deepMF_with_text(object):

    def __init__(self,
                 user_num,
                 item_num,
                 latent_dim,
                 text_latent_dim,                 
                 batch_size,
                 learning_rate,
                 doc_index,
                 doc_index_reverse,
                 doc_mask,
                 doc_mask_bool,
                 word_vec,
                 num_neg,
                 optimizer='adam',
                 dtype=tf.float32,
                 scope='deepMF_with_text'):
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.text_latent_dim=text_latent_dim
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.dtype = dtype
        self.doc_index=doc_index
        self.doc_index_reverse=doc_index_reverse
        self.doc_mask=doc_mask
        self.doc_mask_bool=doc_mask_bool
        self.word_vec=word_vec
        self.word_num=word_vec.shape[0]
        self.num_neg=num_neg
        self.test_shape=[]
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=self.dtype, name='learning_rate')
            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

            self.build_graph()
            #variables = tf.global_variables()
            #for i in xrange(len(variables)):
            #    print variables[i]
            self.saver = tf.train.Saver(max_to_keep=10)


    def build_graph(self):
        self._create_placeholder()
        self._create_embedding()
        self._create_struct()
        self._create_loss()
        #self._create_nn_loss()
        #self._create_multi_class_loss()
        #self._create_BPR_loss()
        #self._create_square_loss()
        self._create_optimizer()
        #for ele in tf.trainable_variables():  
        #    print ele.name  

    def _create_placeholder(self):
        self.users=tf.placeholder(tf.int32,[None],name='users')
        self.pos_items=tf.placeholder(tf.int32,[None],name='pos_items')
        self.neg_item_set=tf.placeholder(tf.int32,[None],name='neg_item_set')
        self.neg_item_index=tf.placeholder(tf.int32,[None,None],name='neg_item_index')

    def _create_embedding(self):
        self.user_embeddings = tf.Variable(tf.random_normal([self.user_num, self.latent_dim], stddev=0.01), dtype=self.dtype,trainable=True, name='user_embeddings')
        self.item_embeddings = tf.Variable(tf.random_normal([self.item_num, self.latent_dim], stddev=0.01), dtype=self.dtype,trainable=True, name='item_embeddings')
        self.doc_index=tf.Variable(self.doc_index,dtype=tf.int32,trainable=False,name='document_index')
        self.doc_index_reverse=tf.Variable(self.doc_index_reverse,dtype=tf.int32,trainable=False,name='document_index_reverse')
        self.doc_mask=tf.Variable(self.doc_mask,dtype=tf.int32,trainable=False,name='document_mask')
        self.doc_mask_bool=tf.Variable(self.doc_mask_bool,dtype=self.dtype,trainable=False,name='document_mask_bool')
        self.word_vec=tf.Variable(self.word_vec,dtype=tf.float32,trainable=True,name='word_vector')
        self.word_prediction_W=tf.Variable(tf.random_normal([self.word_num, self.text_latent_dim], stddev=0.01), dtype=self.dtype,trainable=True, name='word_prediction_W')
        self.word_prediction_b=tf.Variable(tf.random_normal([self.word_num], stddev=0.01), dtype=self.dtype,trainable=True, name='word_prediction_b')
        self.doc_vec_W=tf.Variable(tf.random_normal([3, self.text_latent_dim], stddev=0.01), dtype=tf.float32,trainable=True, name='doc2vec_W')
        self.doc_vec_b=tf.Variable(tf.random_normal([self.text_latent_dim], stddev=0.01), dtype=tf.float32,trainable=True, name='doc2vec_b')
        self.doc_vec_dim_increase=tf.Variable(tf.random_normal([self.text_latent_dim,self.latent_dim], stddev=0.01), dtype=tf.float32,trainable=True, name='increase_dimension')
        #self.doc_vec_attention=tf.Variable(tf.random_normal([20], stddev=0.01), dtype=tf.float32,trainable=True, name='doc2vec_attention')

        #lambda
        #self.lambda_xi=tf.Variable(tf.random_normal([2*self.latent_dim,self.latent_dim], stddev=0.01), dtype=self.dtype,trainable=True, name='lambda_xi')
        #self.lambda_eta=tf.Variable(tf.random_normal([2*self.latent_dim,self.latent_dim], stddev=0.01), dtype=self.dtype,trainable=True, name='lambda_eta')
        #self.lambda1=tf.Variable(tf.random_normal([1], stddev=0.01), dtype=self.dtype,trainable=True, name='lambda_xi')
        #for nn_loss
        #self.nn_loss_W1=tf.Variable(tf.random_normal([self.user_num,self.latent_dim],stddev=0.01),dtype=tf.float32,trainable=True,name='user_neural_network_W1')
        #self.nn_loss_b1=tf.Variable(tf.random_normal([self.user_num,1],stddev=0.01),dtype=tf.float32,trainable=True,name='user_neural_network_b1')
        #self.nn_loss_W2=tf.Variable(tf.random_normal([self.user_num,1,1],stddev=0.01),dtype=tf.float32,trainable=True,name='user_neural_network_W2')
        #self.nn_loss_b2=tf.Variable(tf.random_normal([self.user_num,1],stddev=0.01),dtype=tf.float32,trainable=True,name='user_neural_network_b2')
        #self.nn_loss_item_b=tf.Variable(tf.random_normal([self.item_num,1],stddev=0.01),dtype=tf.float32,trainable=True,name='user_neural_network_item_b')
    def _create_struct(self):
        pos_doc_index=tf.nn.embedding_lookup(self.doc_index,self.pos_items,name='pos_doc_index')
        pos_doc_index_reverse=tf.nn.embedding_lookup(self.doc_index_reverse,self.pos_items,name='pos_doc_index_reverse')
        pos_doc_mask=tf.nn.embedding_lookup(self.doc_mask,self.pos_items,name='pos_doc_mask')
        pos_doc_mask_bool=tf.nn.embedding_lookup(self.doc_mask_bool,self.pos_items,name='pos_doc_mask_bool')

        neg_doc_index=tf.nn.embedding_lookup(self.doc_index,self.neg_item_set,name='neg_doc_index')
        neg_doc_index_reverse=tf.nn.embedding_lookup(self.doc_index_reverse,self.neg_item_set,name='neg_doc_index_reverse')
        neg_doc_mask=tf.nn.embedding_lookup(self.doc_mask,self.neg_item_set,name='neg_doc_mask')
        neg_doc_mask_bool=tf.nn.embedding_lookup(self.doc_mask_bool,self.neg_item_set,name='neg_doc_mask_bool')

        pos_items_doc_vec,self.pos_word_prediction_loss,_=doc2vec.calc_doc_vec(
            doc_index=pos_doc_index,
            doc_index_reverse=pos_doc_index_reverse,
            mask=pos_doc_mask,
            mask_bool=pos_doc_mask_bool,
            word_vec=self.word_vec,
            depth=self.text_latent_dim,
            doc_vec_W=self.doc_vec_W,
            doc_vec_b=self.doc_vec_b,
            dim_increase=self.doc_vec_dim_increase,
            word_prediction_W=self.word_prediction_W,
            word_prediction_b=self.word_prediction_b,
            word_num=self.word_num,
            test_shape=self.test_shape
            )
        neg_items_doc_vec_set,self.neg_word_prediction_loss_set,self.test_shape=doc2vec.calc_doc_vec(
            doc_index=neg_doc_index,
            doc_index_reverse=neg_doc_index_reverse,
            mask=neg_doc_mask,
            mask_bool=neg_doc_mask_bool,
            word_vec=self.word_vec,
            depth=self.text_latent_dim,
            doc_vec_W=self.doc_vec_W,
            doc_vec_b=self.doc_vec_b,
            dim_increase=self.doc_vec_dim_increase,
            word_prediction_W=self.word_prediction_W,
            word_prediction_b=self.word_prediction_b,
            word_num=self.word_num,
            test_shape=self.test_shape
            )
        
        self.user_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users, name='users_embed')
        #get positive embed
        pos_items_embed = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items, name='pos_items_embed')
        #get negative embed
        neg_items_embed_set=tf.nn.embedding_lookup(self.item_embeddings,self.neg_item_set,name='neg_items_embed')

        #get neg items
        neg_items_shape=tf.shape(self.neg_item_index)
        neg_item_index_reshape=tf.reshape(self.neg_item_index,[neg_items_shape[0]*neg_items_shape[1]],name='neg_item_index_reshape')
        neg_items_embed=tf.nn.embedding_lookup(neg_items_embed_set,neg_item_index_reshape,name='neg_items_embed')
        neg_items_doc_vec=tf.nn.embedding_lookup(neg_items_doc_vec_set,neg_item_index_reshape,name='neg_items_doc_vec')
        self.neg_word_prediction_loss=tf.nn.embedding_lookup(self.neg_word_prediction_loss_set,neg_item_index_reshape,name='neg_word_prediction_loss')


        #calculate lambda1
        user_embed_expand=tf.expand_dims(self.user_embed,axis=1)*tf.ones([1,neg_items_shape[1],1])
        user_embed_expand=tf.reshape(user_embed_expand,[-1,self.latent_dim])
        regular=tf.sqrt(tf.cast(self.latent_dim,dtype=tf.float32))
        pos_items_embed_lambda=tf.exp(tf.expand_dims(tf.reduce_sum(self.user_embed*pos_items_embed,axis=1)/regular,axis=1,name='pos_items_embed_lambda'))
        neg_items_embed_lambda=tf.exp(tf.expand_dims(tf.reduce_sum(user_embed_expand*neg_items_embed,axis=1)/regular,axis=1,name='neg_items_embed_lambda'))
        pos_items_doc_lambda=tf.exp(tf.expand_dims(tf.reduce_sum(self.user_embed*pos_items_doc_vec,axis=1)/regular,axis=1,name='pos_items_doc_lambda'))
        neg_items_doc_lambda=tf.exp(tf.expand_dims(tf.reduce_sum(user_embed_expand*neg_items_doc_vec,axis=1)/regular,axis=1,name='neg_items_doc_lambda'))
        
        pos_items_lambda_sum=pos_items_embed_lambda+pos_items_doc_lambda
        neg_items_lambda_sum=neg_items_embed_lambda+neg_items_doc_lambda
        pos_items_embed_lambda/=pos_items_lambda_sum
        neg_items_embed_lambda/=neg_items_lambda_sum
        pos_items_doc_lambda/=pos_items_lambda_sum
        neg_items_doc_lambda/=neg_items_lambda_sum

        self.pos_items_embed=pos_items_embed_lambda*pos_items_embed+pos_items_doc_lambda*pos_items_doc_vec
        neg_items_embed=neg_items_embed_lambda*neg_items_embed+neg_items_doc_lambda*neg_items_doc_vec
        self.neg_items_embed=tf.reshape(neg_items_embed,[neg_items_shape[0],neg_items_shape[1],-1])
        
        #self.pos_items_embed=pos_items_embed
        #self.neg_items_embed=neg_items_embed

    # def _create_struct(self):
    #     doc_index=tf.nn.embedding_lookup(self.doc_index,self.item_inputs,name='doc_index')
    #     doc_index_reverse=tf.nn.embedding_lookup(self.doc_index_reverse,self.item_inputs,name='doc_index_reverse')
    #     doc_mask=tf.nn.embedding_lookup(self.doc_mask,self.item_inputs,name='doc_mask')
    #     doc_mask_bool=tf.nn.embedding_lookup(self.doc_mask_bool,self.item_inputs,name='doc_mask_bool')

    #     #get document vector
    #     item_doc_vec,self.word_prediction_loss,_=doc2vec.calc_doc_vec(
    #         doc_index=doc_index,
    #         doc_index_reverse=doc_index_reverse,
    #         mask=doc_mask,
    #         mask_bool=doc_mask_bool,
    #         word_vec=self.word_vec,
    #         depth=self.text_latent_dim,
    #         doc_vec_W=self.doc_vec_W,
    #         doc_vec_b=self.doc_vec_b,
    #         dim_increase=self.doc_vec_dim_increase,
    #         word_prediction_W=self.word_prediction_W,
    #         word_prediction_b=self.word_prediction_b,
    #         word_num=self.word_num,
    #         test_shape=self.test_shape
    #         )

    #     item_embed=tf.nn.embedding_lookup(self.item_embeddings, self.item_inputs, name='items_embed')
    #     pos_user_embed=tf.nn.embedding_lookup(self.user_embeddings,self.pos_users,name='pos_user_embed')
    #     neg_user_embed=tf.nn.embedding_lookup(self.user_embeddings,self.neg_users,name='neg_user_embed')
    #     user_embed=tf.concat([tf.expand_dims(pos_user_embed,axis=1),neg_user_embed],axis=1)

    #     #calculate lambda1
    #     user_embed_expand=tf.reshape(user_embed,[-1,self.latent_dim])
    #     item_embed_expand=tf.expand_dims(item_embed,axis=1)*tf.ones([1,tf.shape(user_embed)[1],1],tf.float32)
    #     item_embed_expand=tf.reshape(item_embed_expand,[-1,self.latent_dim])
    #     item_doc_vec_expand=tf.expand_dims(item_doc_vec,axis=1)*tf.ones([1,tf.shape(user_embed)[1],1],tf.float32)
    #     item_doc_vec_expand=tf.reshape(item_doc_vec_expand,[-1,self.latent_dim])
    #     #item_embed_expand:[batch_size*(1+num_neg),latent_dim]
    #     #item_doc_vec_expand:[batch_size*(1+num_neg),latent_dim]

    #     regular=tf.sqrt(tf.cast(self.latent_dim,dtype=tf.float32))
    #     item_embed_lambda=tf.exp(tf.expand_dims(tf.reduce_sum(user_embed_expand*item_embed_expand,axis=1)/regular,axis=1,name='item_embed_lambda'))
    #     item_doc_lambda=tf.exp(tf.expand_dims(tf.reduce_sum(user_embed_expand*item_doc_vec_expand,axis=1)/regular,axis=1,name='item_doc_lambda'))
    #     item_lambda_sum=item_embed_lambda+item_doc_lambda
    #     item_embed_lambda=item_embed_lambda/item_lambda_sum
    #     item_doc_lambda=item_doc_lambda/item_lambda_sum
    #     item_embed_lambda=tf.reshape(item_embed_lambda,[tf.shape(item_embed)[0],-1])
    #     item_doc_lambda=tf.reshape(item_doc_lambda,[tf.shape(item_embed)[0],-1])
    #     self.item_vec=tf.expand_dims(item_embed_lambda,axis=2)*tf.expand_dims(item_embed,axis=1)+tf.expand_dims(item_doc_lambda,axis=2)*tf.expand_dims(item_doc_vec,axis=1)
    #     self.user_embed=user_embed

    #     #lambda2
    #     # pos_lambda_embed=tf.tanh(tf.matmul(tf.concat([pos_items_embed,pos_items_doc_vec],axis=1),self.lambda_xi))
    #     # pos_lambda_word=tf.tanh(tf.matmul(tf.concat([pos_items_embed,pos_items_doc_vec],axis=1),self.lambda_eta))
    #     # neg_lambda_embed=tf.tanh(tf.matmul(tf.concat([neg_items_embed,neg_items_doc_vec],axis=1),self.lambda_xi))
    #     # neg_lambda_word=tf.tanh(tf.matmul(tf.concat([neg_items_embed,neg_items_doc_vec],axis=1),self.lambda_eta))
    #     # #self.test_shape.append(self.lambda_xi)
    #     # #self.test_shape.append(self.lambda_eta)
    #     # self.test_shape.append(pos_lambda_embed[0])
    #     # self.test_shape.append(pos_lambda_word[0])


    #     # self.pos_items_embed=pos_lambda_embed*pos_items_embed+pos_lambda_word*pos_items_doc_vec
    #     # neg_items_embed=neg_lambda_embed*neg_items_embed+neg_lambda_word*neg_items_doc_vec

    #     #without lambda
    #     #self.pos_items_embed=0.7*pos_items_embed+0.3*pos_items_doc_vec
    #     #neg_items_embed=0.7*neg_items_embed+0.3*neg_items_doc_vec
        
    #     #self.pos_items_embed=pos_items_embed+pos_items_doc_vec
    #     #neg_items_embed=neg_items_embed+neg_items_doc_vec

    #     #self.neg_items_embed=tf.reshape(neg_items_embed,[neg_items_shape[0],neg_items_shape[1],-1])
    def _create_loss(self):
        self.pos_scores=tf.reduce_sum(self.user_embed*self.pos_items_embed,axis=1)
        #self.pos_scores:[batch_size]

        self.neg_scores=tf.reduce_sum(tf.expand_dims(self.user_embed,axis=1)*self.neg_items_embed,axis=2)
        #self.neg_scores:[batch_size,num_neg]

        self.largest_neg_scores=tf.reduce_max(self.neg_scores,axis=1)
        #self.largest_neg_scores:[batch_size]

        #cross entropy loss
        logits=self.pos_scores-self.largest_neg_scores
        labels=tf.ones(tf.shape(logits),dtype=tf.float32)
        losses= tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        self.loss=tf.reduce_mean(losses)

        #hinge loss
        #losses=tf.nn.relu(1-(self.pos_scores-self.largest_neg_scores))
        #self.loss=tf.reduce_mean(losses)

        #word_prediction_loss
        pos_word_prediction_loss=tf.reduce_mean(self.pos_word_prediction_loss)
        neg_word_prediction_loss=tf.reduce_mean(self.neg_word_prediction_loss)
        word_prediction_loss=pos_word_prediction_loss/(self.num_neg+1)+neg_word_prediction_loss/(self.num_neg+1)*self.num_neg
        self.test_shape.append(self.loss)
        self.test_shape.append(word_prediction_loss)
        self.loss+=0.2*word_prediction_loss
        #self.test_shape.append(self.loss)
    def _create_BPR_loss(self):
        scores=tf.reduce_sum(self.item_vec*self.user_embed,axis=2)
        self.pos_scores=tf.reshape(scores[:,0],[-1])
        #self.pos_scores:[batch_size]

        self.neg_scores=scores[:,1:]
        #self.neg_scores:[batch_size,num_neg]

        self.largest_neg_scores=tf.reduce_max(self.neg_scores,axis=1)
        #self.largest_neg_scores:[batch_size]

        #cross entropy loss
        logits=self.pos_scores-self.largest_neg_scores
        labels=tf.ones(tf.shape(logits),dtype=tf.float32)
        losses= tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        self.loss=tf.reduce_mean(losses)

        #hinge loss
        #losses=tf.nn.relu(1-(self.pos_scores-self.largest_neg_scores))
        #self.loss=tf.reduce_mean(losses)

        #word_prediction_loss
        self.loss+=1*self.word_prediction_loss

    def _get_score(self,user_embed,item_embed):
        #item_embed=item_embed+doc_embed
        user_item=tf.concat([user_embed,item_embed],axis=1,name='concat')
        layer1=tf.layers.dense(user_item,32,activation=tf.nn.relu,name='layer1')
        layer2=tf.layers.dense(layer1,16,activation=tf.nn.relu,name='layer2')
        layer3 = tf.layers.dense(layer2, 8, activation=tf.nn.relu, name='layer3')
        #score = tf.layers.dense(layer3, 1, activation=tf.nn.sigmoid, name='predict')
        score = tf.layers.dense(layer3, 1, activation=tf.nn.sigmoid, name='predict')
        return score
    def _create_multi_class_loss(self):
        pass
        self.multi_class_b=tf.Variable(tf.random_normal([self.item_num], stddev=0.01), dtype=tf.float32,trainable=True, name='multi_class_loss_b')
        losses=tf.nn.sampled_softmax_loss(
                weights=self.item_embeddings,
                biases=self.multi_class_b,
                labels=tf.expand_dims(self.pos_items,axis=1),
                inputs=self.user_embed,
                num_sampled=2000,
                num_classes=self.item_num
                )
        #self.test_shape.append(tf.shape(losses))
        self.loss=tf.reduce_mean(losses)
        self.pos_scores=tf.reduce_sum(self.user_embed*self.pos_items_embed,axis=1)

    def _create_nn_loss(self):
        #put pos_items and neg_items together ->[pos,neg]
        items_embed=tf.concat([tf.expand_dims(self.pos_items_embed,axis=1),self.neg_items_embed],axis=1)
        #items_b=tf.concat([tf.expand_dims(self.pos_items_b,axis=1),self.neg_items_b],axis=1)
        W1 = tf.nn.embedding_lookup(self.nn_loss_W1, self.user_inputs, name='get_neural_network_W1')
        #b1 = tf.nn.embedding_lookup(self.nn_loss_b1, self.user_inputs, name='get_neural_network_b1')
        #W2 = tf.nn.embedding_lookup(self.nn_loss_W2, self.user_inputs, name='get_neural_network_W2')
        #b2 = tf.nn.embedding_lookup(self.nn_loss_b2, self.user_inputs, name='get_neural_network_b2')
        #items:[batch_size,pos+neg,latent_dim]
        #W1:[batch_size,latent_dim]
        scores=tf.reduce_sum(items_embed*tf.expand_dims(W1,axis=1),axis=2)
        #scores:[batch_size,pos+neg]
        #scores+=tf.expand_dims(b1,axis=1)
        #scores+=items_b
        #self.scores=tf.reshape(scores,[tf.shape(scores)[0],tf.shape(scores)[1]])
        self.pos_scores=scores[:,0]
        self.neg_scores=scores[:,1:]
        self.largest_neg_scores=tf.reduce_max(self.neg_scores,axis=1)

    

        #cross entropy loss
        logits=self.pos_scores-self.largest_neg_scores
        labels=tf.ones(tf.shape(logits),dtype=tf.float32)
        losses= tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        self.loss=tf.reduce_mean(losses)

        #hinge loss
        #losses=tf.nn.relu(1-(self.pos_scores-self.largest_neg_scores))
        #self.loss=tf.reduce_mean(losses)
    def _create_square_loss(self):
        pos_doc_index=tf.nn.embedding_lookup(self.doc_index,self.pos_items,name='pos_doc_index')
        pos_doc_mask=tf.nn.embedding_lookup(self.doc_mask,self.pos_items,name='pos_doc_mask')
        neg_doc_index=tf.nn.embedding_lookup(self.doc_index,self.neg_items,name='neg_doc_index')
        neg_doc_mask=tf.nn.embedding_lookup(self.doc_mask,self.neg_items,name='neg_doc_mask')
        pos_doc_vec,self.test_shape=doc2vec.calc_doc_vec(pos_doc_index,pos_doc_mask,self.word_vec,self.doc_vec_dim_increase,self.test_shape)
        neg_doc_vec,_=doc2vec.calc_doc_vec(neg_doc_index,neg_doc_mask,self.word_vec,self.doc_vec_dim_increase,self.test_shape)

        user_embed = tf.nn.embedding_lookup(self.user_embeddings, self.user_inputs, name='users_embed')
        pos_items_embed = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items, name='pos_items_embed')
        neg_items_embed=tf.nn.embedding_lookup(self.item_embeddings,self.neg_items,name='neg_items_embed')

        #pos_items_embed+=pos_doc_vec
        #neg_items_embed+=neg_doc_vec
        self.pos_scores=tf.reduce_sum(user_embed*pos_items_embed,axis=1)
        self.neg_scores=tf.reduce_sum(user_embed*neg_items_embed,axis=1)
        pos_labels=tf.ones(tf.shape(self.pos_scores),tf.float32)
        neg_labels=tf.zeros(tf.shape(self.neg_scores),tf.float32)
        #self.pos_scores=self.pos_scores1+self.pos_scores
        #self.neg_scores=self.neg_scores1+self.neg_scores
        pos_loss=tf.losses.mean_squared_error(labels=pos_labels,predictions=self.pos_scores)
        neg_loss=tf.losses.mean_squared_error(labels=neg_labels,predictions=self.neg_scores)
        self.loss= (pos_loss+neg_loss)/2
        

    def _create_optimizer(self):
        #params = tf.trainable_variables()
        if self.optimizer == 'adam':
            #optimizer = tf.train.AdamOptimizer(self.learning_rate)
            optimizer=tf.contrib.opt.LazyAdamOptimizer(self.learning_rate)
        elif self.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        #self.updates = optimizer2.minimize(self.loss, self.global_step)
        gradients=optimizer.compute_gradients(self.loss)
        #gradients=gradients
        print gradients
        self.grads,variables=zip(*gradients)
        self.grads=list(self.grads)
        self.grads1=[-1*grad for grad in self.grads[12:]]
        self.grads=self.grads[0:12]+self.grads1
        self.gradients=zip(self.grads,variables)
        self.updates=optimizer.apply_gradients(self.gradients,self.global_step)
        # gradients = tf.gradients(self.loss, params)
        # clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        # self.updates = optimizer.apply_gradients(
        #     zip(clipped_gradients, params),
        #     global_step=self.global_step
        # )

    def step(self,session,users,pos_items,neg_item_set,neg_item_index,training):
        input_feed={}
        input_feed[self.users.name]=users
        input_feed[self.pos_items.name]=pos_items
        input_feed[self.neg_item_set.name]=neg_item_set
        input_feed[self.neg_item_index.name]=neg_item_index
        if training: 
          output_feed=[self.loss,self.test_shape,self.updates]
        else:
          output_feed=[self.pos_scores]
        #testing:
        #output_feed=[self.test_shape]
        outputs=session.run(output_feed,input_feed)
        if len(outputs)==1:
            outputs.append(0)
        return outputs[0],outputs[1]
