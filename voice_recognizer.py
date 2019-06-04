import re
import pandas as pd
import speech_recognition as sr

# Information Retrieval
data = pd.read_html('https://en.wikipedia.org/wiki/BBC%27s_100_Greatest_Films_of_the_21st_Century')
# creating a dataframe
df = data[0]
# updating the columns
df.columns = df.iloc[0]
# reindexing
df = df.reindex(df.index.drop(0))
# dropping counter
df.drop('Country', axis=1, inplace=True)

r = sr.Recognizer()

with sr.Microphone() as source:
    print('Please say a Movie Title')
    audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print(f'\nMovie Name: \n{text}')

        text_formatter = ' '.join([x.capitalize() for x in re.findall('[^ ]+', text)])

    except:
        print('Sorry could not recognize your voice.')

print('\n')
print(f'Details of {text_formatter} are ')
print(df[df['Title'] == text_formatter])
