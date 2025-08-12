import ollama
from loadData import dataset
import math

#embedding the data as chunk in the database

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB=[]

def add_chunk_to_database(chunk):
  embedding= ollama.embed(model=EMBEDDING_MODEL,input= chunk) ['embeddings'][0]
  VECTOR_DB.append((chunk,embedding))

for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'Added chunk {i+1}/{len(dataset)} to the database')


#retrieval function

def cosine_similarity(a,b): #ehere a= chunk, b=embedding
  dot_product=sum([x*y for x,y  in zip(a,b)])
  norm_a= math.sqrt(sum([x**2 for x in a]))
  norm_b=math.sqrt(sum([y**2 for y in b]))
  return dot_product/norm_a * norm_b 


#implementing retrieval function 

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0] #similary also embeding the query as embeding vector

  simlarities =[]

  for chunk, embedding in VECTOR_DB:
   similarity=cosine_similarity(query_embedding, embedding)  #calculating similarities between two vectors
   simlarities.append((chunk,similarity))


  simlarities.sort(key= lambda x:x[1] , reverse=True) #listing in the decending order, meaning the most similar/relevent chunks gets sorted
 
#finally we return the N most relevant chunks
  
  return  simlarities[:top_n]


#generating response(chatbot)

input_query = input('What do you wanna know about cat?:')
retrieved_knowledge= retrieve(input_query)


print('Retrieved Knowledge: ')
for chunk,similarity in retrieved_knowledge:
  print(f'-similarity: {similarity:.2f} {chunk}')

#prompt for the chatbot

instruction_prompt=f''' you are a straight-forward chatbot . You use only the  following information provided ,
 do not hallucinate, and do not give out informatinon not present in the  following information provided:
 
{'/n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}

'''

#using ollama for the chatbot to generate responses

stream= ollama.chat(
  model= LANGUAGE_MODEL,
  messages=[
    
      {'role': 'system' , 'content': instruction_prompt},
      {'role' : 'user',   'content': input_query},
  ],
stream=True,
)

# printing the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)






