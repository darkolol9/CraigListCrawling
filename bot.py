import os
import pickle
import discord
import pandas as pd
import numpy as np



TOKEN = 'OTI5NDY3NDIzMzA2OTU2ODEw.Ydnv_g.GAN1pP9F_iFWIJbvtEr1Qg2ks8A'

encoder = pickle.load(open('labelEncoder.pkl','rb'))
model = pickle.load(open('randomForest.pkl','rb'))




def preprocess_data_and_predict(data,model,encoder):
    data =  pd.DataFrame(data)
    encoder.transform(data)
    prediction = model.predict(np.array(data))

    return prediction[0]





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
        res = (preprocess_data_and_predict(data,model,encoder))
        await channel.send(f'*Best i can do* for this car is *slaps rooftop* : {round(res)}$```py\n{data}```')
    
    if message.content.startswith('score'):
        msg = f'```py\n r2_score={82}%```'
        await channel.send(msg)

    if message.content.startswith('howdy'):
        channel = message.channel
        await channel.send('```hoooowdddyyy partner what can i do for you today? ```')


    





client.run(TOKEN)






