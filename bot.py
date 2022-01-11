import os
import pickle
from b import labelEncoder
import discord
import pandas as pd
import numpy as np



TOKEN = 'OTI5NDY3NDIzMzA2OTU2ODEw.Ydnv_g.u_WoRWvQqIABqZ0dtARTyriG1i8'

encoder = pickle.load(open('encoder.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
input_scaler = pickle.load(open('input_scaler.pkl','rb'))
output_scaler = pickle.load(open('output_scaler.pkl','rb'))



# TOKEN = 'token'

# client = discord.Client()

# @client.event
# async def on_ready():
#     print(f'{client.user} has connected to Discord!')

def preprocess_data_and_predict(data,model,scalerX,scalerY,encoder):

    

    data =  pd.DataFrame(data)
    encoder.transform(data)
    data = scalerX.transform(data)
    prediction = model.predict(np.array(data))
    prediction = scalerY.inverse_transform(prediction.reshape(-1,1))

    return prediction[0][0]






data = {'Model':[2010],'Brand':['Toyota'],'Odometer':[146187]
    ,'Paint color':['silver']
    ,'Condition':['good'],'Fuel':['diesel']
    ,'Transmission':['automatic'],'Type':['sedan']}


client = discord.Client()

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):
    if message.content.startswith('check'):
        channel = message.channel
        k = message.content
        data = eval(k[5:])
        res = (preprocess_data_and_predict(data,model,input_scaler,output_scaler,encoder))
        await channel.send(f'*Best i can do* for this car is *slaps rooftop* : {round(res)}$```py\n{data}```')

    if message.content.startswith('howdy'):
        channel = message.channel
        await channel.send('```hoooowdddyyy partner what can i do for you today? ```')


    





client.run(TOKEN)






