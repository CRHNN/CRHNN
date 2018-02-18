import tensorflow as tf
def sent2vec(word_embed,mask,depth):
    shape=tf.shape(word_embed,name='word_embed_shape')
    word_embed_flat=tf.reshape(word_embed,[shape[0]*shape[1],shape[2],shape[3]],name='word_embed_flat')
    mask_flat=tf.reshape(mask,[-1],name='mask_flat')
    with tf.variable_scope("sent2vec"):
        #GRU
        word_embed_flat.set_shape([None,None,depth])
        cell=tf.contrib.rnn.GRUCell(num_units=depth)
        outputs,last_state=tf.nn.dynamic_rnn(cell=cell,inputs=word_embed_flat,sequence_length=mask_flat,dtype=tf.float32)
        #outputs:[batch_size*20,max_time,depth]
        #last_state:[batch_size*20,depth]

        #regularization
        outputs_reg=outputs/5.6#sqrt(32)
        last_state=last_state/5.6#sqrt(32)

        #attention_weight=exp(hl*hL)/Sigma(exp(hl*HL))
        attention=tf.exp(tf.reduce_sum(outputs_reg*tf.expand_dims(last_state,axis=1),axis=2,name='get_sent_attention'))
        #attention:[batch_size*20,10]
        attention_weight=tf.divide(attention,tf.expand_dims(tf.reduce_sum(attention,axis=1),axis=1)+1,name='sent_attention_weight')
        #attention_weight=[batch_size*20,max_time]
        sent_vec=tf.reduce_sum(tf.multiply(outputs,tf.expand_dims(attention_weight,axis=2)),axis=1)
        #sent_vec:[batch_size*20,depth]
        sent_vec=tf.reshape(sent_vec,[shape[0],shape[1],depth],name='get_sent_vec')
        return sent_vec

#vector size:[20,batch_size,depth]
#padded size:[22,batch_size,depth]
def padding(vector):
    shape=tf.shape(vector)
    temp=tf.zeros([1,shape[1],shape[2]])
    padded=tf.concat([temp,vector,temp],axis=0)
    return padded
def conv_while_loop_cond(output,count,input,loop_end,W,b):
    return count<loop_end-1
def conv_while_loop_body(output,count,input,loop_end,W,b):
    start=count-1
    end=start+3
    input_vec=input.gather(tf.range(start,end,dtype=tf.int32))
    output_vec=tf.reduce_sum(tf.expand_dims(W,axis=1)*input_vec,axis=0,name='output_doc_vec')
    output_vec=output_vec+tf.expand_dims(b,axis=0)
    output=output.write(count-1,output_vec)
    count+=1
    return output,count,input,loop_end,W,b
def attention_while_loop_cond(output,attention,h,count,loop_end):
    return count<loop_end
def attention_while_loop_body(output,attention,h,count,loop_end):
    count+=1
    attention=tf.expand_dims(attention,axis=2)
    #attention:[batch_size,max_time,1]
    output=tf.reduce_sum(attention*h,axis=1)
    #output:[batch_size,depth]
    output/=5.6
    h_reg=h/5.6
    attention=tf.exp(tf.reduce_sum(tf.expand_dims(output,axis=1)*h_reg,axis=2))
    #attention:[batch_size,max_time]
    attention=tf.divide(attention,tf.expand_dims(tf.reduce_sum(attention,axis=1),axis=1)+1)
    #attention:[batch_size,max_time]
    return output,attention,h,count,loop_end
def doc2vec(sent_vec,W,b,test_shape):
    #change [batch_size,max_time,depth] to [max_time,batch_size,depth]
    sent_vec=tf.transpose(sent_vec,[1,0,2])
    #padding for sent_vec-->[0,sent_vec,0]
    padded_sent_vec=padding(sent_vec)
    input=tf.TensorArray(tf.float32,size=tf.shape(padded_sent_vec)[0],clear_after_read=False)
    input=input.unstack(padded_sent_vec)

    output=tf.TensorArray(tf.float32,size=tf.shape(sent_vec)[0])
    loop_end=tf.shape(padded_sent_vec)[0]
    count=1
    output,_,_,_,_,_=tf.while_loop(cond=conv_while_loop_cond,body=conv_while_loop_body,loop_vars=[output,count,input,loop_end,W,b])
    h=output.stack()
    #change [max_time,batch_size,depth] to [batch_size,max_time,depth]
    h=tf.transpose(h,[1,0,2])
    #test_shape.append(tf.shape(h))
    #return h,test_shape
    #attention
    shape=tf.shape(h)
    attention=tf.ones([shape[0],shape[1]],dtype=tf.float32)/tf.cast(shape[1],tf.float32)
    #attention:[batch_size,max_time]
    output=tf.ones([shape[0],shape[2]],dtype=tf.float32)
    loop_end=15
    count=1
    _,attention,h,_,_=tf.while_loop(cond=attention_while_loop_cond,body=attention_while_loop_body,loop_vars=[output,attention,h,count,loop_end])
    attention=tf.expand_dims(attention,axis=2)
    output=tf.reduce_sum(attention*h,axis=1)
    return output,test_shape

def word_prediction(sent_vec,mask,mask_bool,depth,input_shape,doc_index_reverse,word_prediction_W,word_prediction_b,word_num,test_shape):
    sent_shape=tf.shape(sent_vec,name='sent_vec_shape')
    sent_embed_flat=tf.reshape(sent_vec,[sent_shape[0]*sent_shape[1],sent_shape[2]],name='sent_vec_flat')
    mask_flat=tf.reshape(mask,[-1],name='mask_flat_1')
    doc_index_reverse_flat=tf.reshape(doc_index_reverse,[input_shape[0]*input_shape[1]*input_shape[2]],name='doc_index_reverse_flat')
    #GRU
    cell=tf.contrib.rnn.GRUCell(num_units=depth)
    zeros=tf.zeros((input_shape[0]*input_shape[1],input_shape[2],input_shape[3]))
    zeros.set_shape([None,None,depth])
    sent_embed_flat.set_shape([None,depth])
    outputs,last_state=tf.nn.dynamic_rnn(cell=cell,initial_state=sent_embed_flat,inputs=zeros,sequence_length=mask_flat,dtype=tf.float32)
    #outputs:[batch_size*max_sent,max_word,depth]
    outputs_flat=tf.reshape(outputs,[input_shape[0]*input_shape[1]*input_shape[2],input_shape[3]],name='word_pre_index_flat')
    losses=tf.nn.sampled_softmax_loss(
        weights=word_prediction_W,
        biases=word_prediction_b,
        labels=tf.expand_dims(doc_index_reverse_flat,axis=1),
        inputs=outputs_flat,
        num_sampled=200,
        num_classes=word_num
        )
    #losses:[batch_size*max_sent*max_word]
    losses=tf.reshape(losses,[input_shape[0],input_shape[1],input_shape[2]],name='loss_reshape')

    loss=tf.reduce_sum((losses*mask_bool)/(tf.expand_dims(tf.expand_dims(tf.reduce_sum(mask_bool,axis=[1,2]),axis=1),axis=1)+0.01) ,axis=[1,2],name='word_prediction_loss')
    #loss=tf.reduce_sum((losses*mask_bool)/tf.reduce_sum(mask_bool) ,axis=[1,2],name='word_prediction_loss')
    #loss=tf.reduce_mean(losses)
    return loss,test_shape
def calc_doc_vec(doc_index,doc_index_reverse,mask,mask_bool,word_vec,depth,doc_vec_W,doc_vec_b,
                 dim_increase,word_prediction_W,word_prediction_b,word_num,test_shape):
    word_embed= tf.nn.embedding_lookup(word_vec, doc_index, name='word_embed')
    #test_shape.append(word_embed)
    sent_vec=sent2vec(word_embed,mask,depth)
    #sent_vec:[batch_size,20,depth]
    doc_vec,test_shape=doc2vec(sent_vec,doc_vec_W,doc_vec_b,test_shape)
    doc_vec=tf.reduce_sum(tf.expand_dims(doc_vec,axis=2)*tf.expand_dims(dim_increase,axis=0),axis=1)
    word_prediction_loss,test_shape=word_prediction(sent_vec,mask,mask_bool,depth,tf.shape(word_embed),doc_index_reverse,word_prediction_W,word_prediction_b,word_num,test_shape)
    #word_prediction_loss=tf.zeros([128])
    return doc_vec,word_prediction_loss,test_shape
