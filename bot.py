import os
import pickle
import discord
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import requests



##############################
# made by darkolol9 #
#############################



TOKEN = 'OTI5NDY3NDIzMzA2OTU2ODEw.Ydnv_g.GAN1pP9F_iFWIJbvtEr1Qg2ks8A'

encoder = pickle.load(open('encoder.pkl','rb'))
model = pickle.load(open('randomForest.pkl','rb'))


def url_to_data_dict(url:str) -> dict:
  data = {'Model':0,
          'Brand':0,
          'Odometer':0,
          'Paint color':0,
          'Condition':0,
          'Fuel':0,
          'Transmission':0,
          'Type':0}





  stats = {}

  soup = BeautifulSoup(requests.get(url).content,'lxml')

  headline = soup.find('span',attrs={'id':'titletextonly'})

  data['Model'] = re.findall('\d\d\d\d',headline.text)[0]
  data['Brand'] = re.findall(' ([A-Za-z]+)',headline.text)[0].upper()

  features = soup.find_all('p',class_='attrgroup')[1]


  for sp in features:
    try:
      stats[sp.text[:sp.text.find(':')]] = sp.find('b').text
    except:
      pass



  for key in stats:
    if key.title() in data:
      data[key.title()] = stats[key]

    if key == 'paint color':
      data['Paint color'] = stats[key]


  return str(data)


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
        data = k[5:]
        data = url_to_data_dict(data)
        res = (preprocess_data_and_predict(eval(data),model,encoder))
        await channel.send(f'*Best i can do* for this car is *slaps rooftop* : {round(res)}$```py\n{data}```')
        # await channel.send(data)
    
    if message.content.startswith('score'):
        msg = f'```py\n r2_score={82}%```'
        await channel.send(msg)

    if message.content.startswith('howdy'):
        channel = message.channel
        await channel.send('```hoooowdddyyy partner what can i do for you today? ```')


    





client.run(TOKEN)






